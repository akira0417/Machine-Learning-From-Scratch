from utils import load_breast_cancer_data
import numpy as np
from logistic_regression import sigmoid

predictions = [0.99, 0.9, 0.7, 0.5, 0.2, 0.01]

print("假設真實答案 y = 1")

for p in predictions:
    loss = -np.log(p)
    print(f"Prediction: {p:.2f}, Loss: {loss:.4f}")


'''
df = load_breast_cancer_data("2_LogisticRegression/data/wdbc.data")

df["diagnosis"] = df["diagnosis"].map({
    "B":0,
    "M":1
})

x = df["radius_mean"].to_numpy(dtype=float)

y = df["diagnosis"].to_numpy(dtype=float)

x_mean = x.mean()
x_std = x.std()

x = (x - x_mean) / x_std

# 固定亂數種子，讓每次結果都一樣
np.random.seed(42)

# 產生一個隨機排列的索引
indices = np.random.permutation(len(x))

# 同時打亂 x 和 y
x = x[indices]
y = y[indices]

split_index = int(len(x) * 0.8)

x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

'''