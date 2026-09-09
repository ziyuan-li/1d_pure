from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

# The source train.csv contains the full dataset.
df = pd.read_csv(Path(__file__).resolve().parent / "data" / "train.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  # Critical temperature target.

# Reserve 20% of the rows for testing.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Reserve 20% of the remaining rows for validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=0
)
