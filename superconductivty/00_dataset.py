from sklearn.model_selection import train_test_split
import pandas as pd

# 读取您上传的 train.csv (它其实是全量数据)
df = pd.read_csv("./data/train.csv")

X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values  # 标签 (critical_temp)

# === 关键步骤：手动切分出测试集 ===
# 1. 切出 20% 作为 Test (这个就是您的测试集)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# 2. 从剩余数据中切出 Val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=0
)