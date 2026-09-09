"""Train real-valued models for sample-efficiency experiments."""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pure_resmlp.models.mlp import build_4layer_mlp
from pure_resmlp.models.pure import RealPUResNet1D
from pure_resmlp.paths import DATA_DIR, EFFICIENCY_REAL_CHECKPOINT_DIR


def load_data_raw(dataset_name):
    """Load unscaled features and targets."""
    if dataset_name == "friedman":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)
    elif dataset_name == "concrete":
        df = pd.read_excel(DATA_DIR / "concrete+compressive+strength" / "Concrete_Data.xls")
        X = df.iloc[:, :8].values
        y = df.iloc[:, 8].values
    else:  # california
        data = fetch_california_housing()
        X, y = data.data, data.target
    return X, y


def get_data_splits(X, y, fraction, seed):
    """Split rows 64/16/20 with seed 0, then subsample training rows.

    The run seed selects the training subset; validation and test rows stay fixed.
    """
    # Split seed 0 fixes the test partition across runs.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Validation uses 20% of the remaining rows.
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=0
    )

    if fraction < 1.0:
        n_samples = int(X_train_full.shape[0] * fraction)
        # The run seed controls training-subset selection.
        rng = np.random.RandomState(seed)
        indices = rng.choice(X_train_full.shape[0], n_samples, replace=False)
        X_train_sub = X_train_full[indices]
        y_train_sub = y_train_full[indices]
    else:
        X_train_sub = X_train_full
        y_train_sub = y_train_full

    return X_train_sub, X_val_full, X_test, y_train_sub, y_val_full, y_test


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / n


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="concrete", choices=["friedman", "concrete", "california"])
    parser.add_argument("--fraction", type=float, default=0.8, help="训练数据比例 (0.1 - 1.0)")
    parser.add_argument("--runs", type=int, default=5, help="重复训练次数")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=str(EFFICIENCY_REAL_CHECKPOINT_DIR))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Training: Dataset={args.dataset} | Fraction={args.fraction} | Runs={args.runs}")

    X_raw, y_raw = load_data_raw(args.dataset)
    input_dim = X_raw.shape[1]

    for run in range(args.runs):
        (X_train, X_val, X_test,
         y_train, y_val, y_test) = get_data_splits(X_raw, y_raw, args.fraction, seed=run)

        # Fit both scalers on the selected training rows.
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train_s = x_scaler.transform(X_train)
        X_val_s = x_scaler.transform(X_val)
        X_test_s = x_scaler.transform(X_test)

        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        train_ds = TensorDataset(torch.tensor(X_train_s).float(), torch.tensor(y_train_s).float().unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_s).float(), torch.tensor(y_val_s).float().unsqueeze(1))
        test_ds = TensorDataset(torch.tensor(X_test_s).float(), torch.tensor(y_test_s).float().unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        models_to_train = [
            ("mlp", "mlp4_h128", lambda: build_4layer_mlp(input_dim, 1, 128)),
            ("pu", "puresnet_h128", lambda: RealPUResNet1D(input_dim, 128, 1))
        ]

        for m_type, m_tag, m_builder in models_to_train:
            model = m_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            best_state = None

            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = eval_one_epoch(model, val_loader, criterion, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

            if best_state is not None:
                model.load_state_dict(best_state)

            test_loss = eval_one_epoch(model, test_loader, criterion, device)

            save_name = f"{m_tag}_{args.dataset}_frac{args.fraction}_run{run}_best.pth"
            save_path = os.path.join(args.save_dir, save_name)

            torch.save({
                "model_state_dict": best_state,
                "input_dim": input_dim,
                "hidden_dim": 128,
                "fraction": args.fraction,
                "test_mse": test_loss,
                "x_scaler_mean": x_scaler.mean_,
                "x_scaler_scale": x_scaler.scale_,
                "y_scaler_mean": y_scaler.mean_,
                "y_scaler_scale": y_scaler.scale_
            }, save_path)

            print(f"  [Run {run}][{m_type}] Best Val: {best_val_loss:.4f} | Test MSE: {test_loss:.4f} -> Saved.")


if __name__ == "__main__":
    main()
