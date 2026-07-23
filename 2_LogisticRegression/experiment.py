from utils import load_breast_cancer_data
import numpy as np
from logistic_regression import LogisticRegression
from metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1, std)

    X_Scaled = (X - mean) / std

    return X_Scaled

df = load_breast_cancer_data("2_LogisticRegression/data/wdbc.data")

df["diagnosis"] = df["diagnosis"].map({
    "B":0,
    "M":1
})

x = df.iloc[:, 2:].to_numpy(dtype=float)
y = df["diagnosis"].to_numpy(dtype=float)

x = standardize(x)

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


print("Logistic Regression with Different Learning Rates")
learning_rate = [0.1, 0.01, 0.001]

for lr in learning_rate:
    print(f"\nLearning Rate: {lr}")
    model = LogisticRegression(
        learning_rate=lr,
        epochs=500
    )

    model.fit(x_train, y_train)


    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print("\n訓練結果")
    print(f"Train Accuracy : {accuracy_score(y_train, train_pred):.6f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred):.6f}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")

print("\nLogistic Regression with Different Epochs")
epochs = [100, 500, 1000]

for ep in epochs:
    print(f"\nEpochs: {ep}")
    model = LogisticRegression(
        learning_rate=0.01,
        epochs=ep
    )

    model.fit(x_train, y_train)


    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print("\n訓練結果")
    print(f"Train Accuracy : {accuracy_score(y_train, train_pred):.6f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred):.6f}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")