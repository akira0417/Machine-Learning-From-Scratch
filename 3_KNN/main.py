import numpy as np
from utils import standardize, train_test_split
from data_loader import load_breast_cancer_data
from knn import KNN
from metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = load_breast_cancer_data("2_LogisticRegression/data/wdbc.data")

df["diagnosis"] = df["diagnosis"].map({
    "B":0,
    "M":1
})

x = df.iloc[:, 2:].to_numpy(dtype=float)
y = df["diagnosis"].to_numpy(dtype=int)

X = standardize(x)

# 固定亂數種子，讓每次結果都一樣
np.random.seed(42)

# 產生一個隨機排列的索引
indices = np.random.permutation(len(X))

# 同時打亂 X 和 y
X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = KNN(k=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))