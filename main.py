from utils import load_auto_mpg, clean_data, loss_curve, regression_line
from linear_regression import gradient_descent


df = load_auto_mpg("data/auto-mpg.data")

df = clean_data(df)

x = df["weight"].to_numpy(dtype=float)
y = df["mpg"].to_numpy(dtype=float)

x_mean = x.mean()
x_std = x.std()
x_scaled = (x - x_mean) / x_std

w = 50
b = 50

w, b, history = gradient_descent(x_scaled, y, w, b, learning_rate=0.01, epochs=10000)

loss_curve(history)
regression_line(x_scaled, y, w, b)

print(f"Final parameters: w = {w}, b = {b}")