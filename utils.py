import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import predict
import numpy as np

def load_auto_mpg(file_path):
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    df = pd.read_csv(file_path, sep=r'\s+', names=columns, na_values='?', quotechar='"', engine="c")
    return df

def clean_data(df): # drop rows with missing values
    df = df.dropna()
    return df

def loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], color='blue', linewidth=2)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

def regression_line(x, y, w, b):
    y_pred = predict(x, w, b)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='steelblue', label='Data', alpha=0.6)
    sorted_index = np.argsort(x)
    x_sorted = x[sorted_index]
    y_sorted = y_pred[sorted_index]
    plt.plot(x_sorted, y_sorted, color='red', linewidth=2, label='Regression Line')
    plt.xlabel("Weight")
    plt.ylabel("MPG")
    plt.title("Linear Regression: Weight vs MPG")
    plt.grid(True)
    plt.legend()
    plt.show()