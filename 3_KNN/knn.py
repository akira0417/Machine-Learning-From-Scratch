import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict(self, x):
        distances = []

        for x_train in self.X_train:
            distance = euclidean_distance(x, x_train)
            distances.append(distance)

        k_indices = np.argsort(distances)[:self.k]

        k_labels = self.y_train[k_indices]

        prediction = np.bincount(k_labels).argmax()

        return prediction

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))