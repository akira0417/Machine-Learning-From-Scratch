import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, w, b):
    z = w * x + b
    return sigmoid(z)

def compute_loss(y_true, y_pred):
    """
    Binary Cross Entropy Loss
    """

    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return np.mean(loss)

print(np.log(1))
print(np.log(0.5))
print(np.log(0))