import numpy as np

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1, std)

    X_Scaled = (X - mean) / std

    return X_Scaled

def train_test_split(X, y, test_size, random_state):
    # 固定亂數種子，讓每次結果都一樣
    np.random.seed(random_state)

    # 產生一個隨機排列的索引
    indices = np.random.permutation(len(X))

    # 同時打亂 X 和 y
    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test