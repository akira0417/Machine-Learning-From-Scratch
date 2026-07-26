import numpy as np

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1, std)

    X_Scaled = (X - mean) / std

    return X_Scaled

def train_test_split(x, y):
    split_index = int(len(x) * 0.8)

    x_train = x[:split_index]
    y_train = y[:split_index]

    x_test = x[split_index:]
    y_test = y[split_index:]

    return x_train, x_test, y_train, y_test