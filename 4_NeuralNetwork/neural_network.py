import numpy as np

from activations import relu, relu_derivative, sigmoid
from losses import binary_cross_entropy


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        learning_rate=0.01,
        epochs=1000
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 第一層參數
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)

        # 第二層參數
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)

        self.loss_history = []

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2
    
    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = self.A2 - y

        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)

        dW1 = (X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0)

        return dW1, db1, dW2, db2
    
    def update(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, y):
        self.loss_history = []

        for epoch in range(self.epochs):

            # Forward
            y_pred = self.forward(X)

            # Loss
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Backward
            dW1, db1, dW2, db2 = self.backward(X, y)

            # Update
            self.update(dW1, db1, dW2, db2)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Loss: {loss:.6f}"
                )

    def predict_proba(self, X):
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)