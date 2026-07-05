def predict(x, w, b): # linear regression
    return w * x + b

def loss(y, y_pred): # MSE
    return ((y - y_pred) ** 2).mean()

def gradient(x, y, y_pred): # gradient of MSE
    n = len(x)
    dw = (2/n) * (x * (y_pred - y)).sum()
    db = (2/n) * (y_pred - y).sum()
    return dw, db

def gradient_descent(x, y, w, b, learning_rate, epochs):
    history = {
        "loss": [],
        "w": [],
        "b": []
    }
    for epoch in range(epochs):
        y_pred = predict(x, w, b)
        l = loss(y, y_pred)
        dw, db = gradient(x, y, y_pred)
        w -= learning_rate * dw
        b -= learning_rate * db
        history["loss"].append(l)
        history["w"].append(w)
        history["b"].append(b)  
        if (epoch+1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {l}")

    return w, b, history