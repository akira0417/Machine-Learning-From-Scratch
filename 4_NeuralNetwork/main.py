import matplotlib.pyplot as plt
from data_loader import load_data
from utils import train_test_split, standardize
from neural_network import NeuralNetwork
from metrics import accuracy_score

X, y = load_data("4_NeuralNetwork/data/wdbc.data")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = standardize(X_train)
X_test = standardize(X_test)

model = NeuralNetwork(
    input_size=30,
    hidden_size=8,
    learning_rate=0.01,
    epochs=5000
)

model.fit(
    X_train,
    y_train
)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(
    y_train.flatten(),
    train_pred.flatten()
)

test_accuracy = accuracy_score(
    y_test.flatten(),
    test_pred.flatten()
)

print("\nTraining Result")

print(
    f"Train accuracy: {train_accuracy:.4f}"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)

print(
    f"Final loss: {model.loss_history[-1]:.6f}"
)

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy")
plt.title("Neural Network Training Loss")

plt.grid(True)
plt.show()