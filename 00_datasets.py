""" -----------------------------
    1. Friedman Synthetic Dataset
    Train / Val / Test = 70 / 15 / 15
-----------------------------------"""

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X_1, y_1 = make_friedman1(n_samples=10000, n_features=10, noise=0.0, random_state=0)

# 70% train, 30% temp
X1_train, X1_temp, y1_train, y1_temp = train_test_split(
    X_1, y_1, test_size=0.30, random_state=0
)

# From the 30% → 15% val + 15% test
X1_val, X1_test, y1_val, y1_test = train_test_split(
    X1_temp, y1_temp, test_size=0.50, random_state=0
)



""" -----------------------------
    2. Concrete Strength Dataset
    Train / Val / Test = 60 / 20 / 20
-----------------------------------"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("./data/concrete+compressive+strength/Concrete_Data.xls")

X_2 = df.iloc[:, :8].values
y_2 = df.iloc[:, 8].values

# 60% train, 40% temp
X2_train, X2_temp, y2_train, y2_temp = train_test_split(
    X_2, y_2, test_size=0.40, random_state=0
)

# From the 40% → 20% val + 20% test
X2_val, X2_test, y2_val, y2_test = train_test_split(
    X2_temp, y2_temp, test_size=0.50, random_state=0
)



""" -----------------------------
    3. California Housing Dataset
    Train / Val / Test = 80 / 10 / 10
-----------------------------------"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X_3, y_3 = data.data, data.target

# 80% train, 20% temp
X3_train, X3_temp, y3_train, y3_temp = train_test_split(
    X_3, y_3, test_size=0.20, random_state=0
)

# 20% → 10% val + 10% test
X3_val, X3_test, y3_val, y3_test = train_test_split(
    X3_temp, y3_temp, test_size=0.50, random_state=0
)



""" -----------------------------
    print
-----------------------------------"""

print("Friedman:", X1_train.shape, X1_val.shape, X1_test.shape)
print("Concrete:", X2_train.shape, X2_val.shape, X2_test.shape)
print("California:", X3_train.shape, X3_val.shape, X3_test.shape)
