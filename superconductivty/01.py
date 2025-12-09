# 11_train_superconduct.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 引入您的 PU 模型
# 确保 pu_models.py 在同一目录下
from pu_models import RealPUResNet1D


def set_seed(seed=42):
    """固定随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_superconductivity_data(data_dir="./data"):
    """
    加载 Superconductivity 数据集
    逻辑：
    1. 读取 train.csv (全量数据)
    2. 切分 Test (20%)
    3. 从剩余部分切分 Val (20%) -> 最终 Train 占比 64%
    """
    # 兼容文件名可能是 train.csv 或 superconduct.csv
    paths = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "superconduct.csv")
    ]

    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError(f"在 {data_dir} 下找不到 train.csv 或 superconduct.csv")

    print(f"正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)

    # 81个特征 + 1个Target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 1. 划分 Test Set (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 2. 划分 Validation Set (从剩余的 80% 中再切 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=42
    )

    print(f"数据划分完成:")
    print(f"  Train: {X_train.shape[0]} 样本")
    print(f"  Val  : {X_val.shape[0]} 样本")
    print(f"  Test : {X_test.shape[0]} 样本")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128):
    """标准化数据并封装为 PyTorch DataLoader"""
    # 标准化 (fit on Train only)
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    # Target 标准化 (这对回归任务收敛很重要)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # 转 Tensor
    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                             torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                           torch.tensor(y_val_s, dtype=torch.float32).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32),
                            torch.tensor(y_test_s, dtype=torch.float32).unsqueeze(1))

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader, y_scaler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        count += X.size(0)
    return total_loss / count


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total_loss += loss.item() * X.size(0)
            count += X.size(0)
    return total_loss / count


def evaluate_rmse_original_scale(model, loader, device, y_scaler):
    """在原始物理尺度 (Kelvin) 上计算 RMSE"""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            # 预测值 (Standardized)
            pred_s = model(X).cpu().numpy()
            # 真实值 (Standardized)
            y_s = y.cpu().numpy()

            # 反标准化
            preds.append(y_scaler.inverse_transform(pred_s))
            targets.append(y_scaler.inverse_transform(y_s))

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128, help="PU网络神经元数量")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_superconduct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_superconductivity_data()
    input_dim = X_train.shape[1]

    # 2. 制作 Loader
    train_loader, val_loader, test_loader, y_scaler = make_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=args.batch_size
    )

    # 3. 初始化模型 (PU-ResNet)
    print(f"初始化 RealPUResNet1D (Hidden={args.hidden_dim})...")
    model = RealPUResNet1D(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # 在标准化空间计算 MSE

    # 4. 训练循环
    best_val_loss = float('inf')
    best_ckpt_path = os.path.join(args.save_dir, "best_pu_superconduct.pth")

    print(f"开始训练 (Epochs={args.epochs})...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, test_loader, criterion, device)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'hidden_dim': args.hidden_dim,
                'input_dim': input_dim
            }, best_ckpt_path)
            # 可选：打印个星号标记
            mark = "*"
        else:
            mark = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {mark}")

    print(f"\n训练结束。最佳验证集 Loss: {best_val_loss:.5f}")
    print(f"最佳模型已保存至: {best_ckpt_path}")

    # 5. 加载最佳模型并在测试集评估
    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 计算 RMSE (物理单位 K)
    test_rmse = evaluate_rmse_original_scale(model, test_loader, device, y_scaler)

    print("=" * 40)
    print(f"Final Test Result (Superconductivity):")
    print(f"RMSE: {test_rmse:.4f} K")
    print("=" * 40)


if __name__ == "__main__":
    main()