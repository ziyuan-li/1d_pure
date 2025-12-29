# 01.py
# Train script (Superconductivity regression) with multiple selectable networks.
# - Keep dataset split reproducible via --split_seed
# - Remove model initialization seeding (weights differ each run)

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from mlp_models_superconductivty import build_4layer_mlp
from pu_models_superconductivty import RealPUResNet1D
from complex_pu_model_superconductivty import ComplexPUResNet1D
from complex_mlp_models_superconductivty import build_complex_mlp


def load_superconductivity_data(data_dir: str = "./data", split_seed: int = 42):
    """
    Load Superconductivity dataset and do a FIXED split with split_seed.

    Logic:
      1) Read train.csv (or superconduct.csv)
      2) Split Test (20%)
      3) Split Val from remaining (20% of 80%) -> Train is 64%

    Note:
      - This split is reproducible across runs due to random_state=split_seed.
      - We intentionally do NOT set torch/np seeds for model initialization.
    """
    paths = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "superconduct.csv"),
    ]

    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError(f"Cannot find train.csv or superconduct.csv under: {data_dir}")

    print(f"Loading data: {file_path} ...")
    df = pd.read_csv(file_path)

    # 81 features + 1 target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 1) Test split (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=split_seed
    )

    # 2) Val split (20% of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=split_seed
    )

    print("Split done:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val  : {X_val.shape[0]} samples")
    print(f"  Test : {X_test.shape[0]} samples")
    print(f"  split_seed: {split_seed}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int = 128
):
    """Standardize X/y and wrap into PyTorch DataLoaders."""
    # X standardization (fit on train only)
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    # y standardization (very helpful for regression convergence)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val_s, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test_s, dtype=torch.float32).unsqueeze(1),
    )

    # loaders
    # NOTE: shuffle=True will be non-deterministic across runs (intended),
    # because we removed global seeds for model initialization.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader, y_scaler


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, input_dim: int, hidden_dim: int, output_dim: int = 1) -> nn.Module:
    """
    Factory to build different architectures.

    model_name choices:
      - mlp
      - real_pu
      - complex_mlp
      - complex_pu
    """
    if model_name == "mlp":
        print("Model: Real-valued MLP (build_4layer_mlp)")
        return build_4layer_mlp(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    if model_name == "real_pu":
        print("Model: Real-valued PU-ResNet (RealPUResNet1D)")
        return RealPUResNet1D(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    if model_name == "complex_mlp":
        print("Model: Complex-valued MLP (build_complex_mlp)")
        return build_complex_mlp(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    if model_name == "complex_pu":
        print("Model: Complex-valued PU-ResNet (ComplexPUResNet1D)")
        return ComplexPUResNet1D(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    raise ValueError(f"Unknown model: {model_name}")


def _maybe_to_complex_input(X: torch.Tensor, is_complex_model: bool) -> torch.Tensor:
    """Promote real input to complex for complex-valued networks."""
    if not is_complex_model:
        return X
    if torch.is_complex(X):
        return X
    return torch.complex(X, torch.zeros_like(X))


def _pred_to_real(pred: torch.Tensor) -> torch.Tensor:
    """Targets are real-valued; if model outputs complex, use real part."""
    return pred.real if torch.is_complex(pred) else pred


def train_one_epoch(model, loader, criterion, optimizer, device, is_complex_model: bool):
    model.train()
    total_loss = 0.0
    count = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = _maybe_to_complex_input(X, is_complex_model)

        optimizer.zero_grad()
        pred = model(X)
        pred_r = _pred_to_real(pred)

        loss = criterion(pred_r, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * X.size(0)
        count += X.size(0)

    return total_loss / max(count, 1)


def eval_one_epoch(model, loader, criterion, device, is_complex_model: bool):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            X = _maybe_to_complex_input(X, is_complex_model)

            pred = model(X)
            pred_r = _pred_to_real(pred)
            loss = criterion(pred_r, y)

            total_loss += float(loss.item()) * X.size(0)
            count += X.size(0)

    return total_loss / max(count, 1)


def evaluate_rmse_original_scale(model, loader, device, y_scaler, is_complex_model: bool):
    """Compute RMSE on original physical scale (Kelvin)."""
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            X = _maybe_to_complex_input(X, is_complex_model)

            pred_s = _pred_to_real(model(X)).cpu().numpy()  # standardized
            y_s = y.cpu().numpy()                           # standardized

            preds.append(y_scaler.inverse_transform(pred_s))
            targets.append(y_scaler.inverse_transform(y_s))

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dim for networks")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_superconduct")

    # Keep ONLY the split seed
    parser.add_argument("--split_seed", type=int, default=42, help="seed for dataset split only")

    # Choose model
    parser.add_argument(
        "--model",
        type=str,
        default="complex_pu",
        choices=["mlp", "real_pu", "complex_mlp", "complex_pu"],
        help="choose network architecture",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Note: model init is NOT seeded; weights will differ each run.")
    print("Note: dataset split IS fixed via --split_seed.")

    is_complex_model = args.model in ["complex_mlp", "complex_pu"]

    # 1) load data with fixed split seed
    X_train, X_val, X_test, y_train, y_val, y_test = load_superconductivity_data(
        data_dir=args.data_dir,
        split_seed=args.split_seed,
    )
    input_dim = X_train.shape[1]

    # 2) loaders
    train_loader, val_loader, test_loader, y_scaler = make_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=args.batch_size
    )

    # 3) build model (unseeded init)
    model = build_model(
        model_name=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
    ).to(device)

    print(f"Trainable params: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # MSE in standardized space

    # 4) training loop
    best_val_loss = float("inf")
    ckpt_name = f"best_{args.model}_hidden{args.hidden_dim}_split{args.split_seed}.pth"
    best_ckpt_path = os.path.join(args.save_dir, ckpt_name)

    print(f"Start training (epochs={args.epochs}, model={args.model}, split_seed={args.split_seed}) ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, is_complex_model)
        val_loss = eval_one_epoch(model, test_loader, criterion, device, is_complex_model)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": args.model,
                    "hidden_dim": args.hidden_dim,
                    "input_dim": input_dim,
                    "val_loss": val_loss,
                    "split_seed": args.split_seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_ckpt_path,
            )

        if epoch % 10 == 0 or epoch == 1:
            mark = "*" if improved else ""
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} {mark}")

    print(f"\nTraining done. Best Val Loss: {best_val_loss:.6f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")

    # 5) load best model and evaluate on test set (RMSE in Kelvin)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_rmse = evaluate_rmse_original_scale(model, test_loader, device, y_scaler, is_complex_model)

    print("=" * 50)
    print("Final Test Result (Superconductivity):")
    print(f"Model: {args.model} | hidden_dim={args.hidden_dim} | split_seed={args.split_seed}")
    print(f"RMSE: {test_rmse:.4f} K")
    print("=" * 50)


if __name__ == "__main__":
    main()
