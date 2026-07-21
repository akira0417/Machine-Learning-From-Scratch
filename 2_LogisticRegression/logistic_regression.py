import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Parameters
        ----------
        learning_rate : float
            每次更新參數時的步長。

        epochs : int
            使用全部訓練資料更新參數的次數。
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """
        將任意實數轉換成 0 到 1 之間的機率。
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """
        計算 Binary Cross Entropy。
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return np.mean(loss)

    def fit(self, X, y):
        """
        使用 Gradient Descent 訓練模型。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X 必須是二維陣列，形狀應為 (樣本數, 特徵數)。")

        if y.ndim != 1:
            y = y.reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的樣本數必須相同。")

        n_samples, n_features = X.shape

        # 一個特徵對應一個權重
        self.weights = np.zeros(n_features)

        # bias 只有一個
        self.bias = 0.0

        # 避免重複呼叫 fit() 時保留舊紀錄
        self.loss_history = []

        for epoch in range(self.epochs):
            # Forward propagation
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # 計算目前 Loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward propagation
            error = y_pred - y

            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Gradient Descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {loss:.6f}"
                )

        return self

    def predict_proba(self, X):
        """
        回傳每筆資料屬於類別 1 的機率。
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        根據 threshold 將機率轉換成 0 或 1。
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def _check_is_fitted(self):
        """
        確認模型是否已經完成訓練。
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
