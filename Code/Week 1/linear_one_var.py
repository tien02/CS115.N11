import time
import numpy as np
import matplotlib.pyplot as plt

def create_dummy_dataset(len_data=100):
    w = np.random.randint(20, size=1)[0]
    b = np.random.randint(20, size=1)[0]
    X = np.arange(1, len_data + 1)
    y = X * w + b + np.random.randint(-100, 100, size=X.shape)
    return X, y

def MSELoss(y, y_pred):
    return np.mean(np.square(y - y_pred))

def inference(x, w, b):
    return w * x + b

def w_grad(x, y, y_pred):
    return -2 * np.mean((y-y_pred) * x)

def b_grad(y, y_pred):
    return -2 * np.mean(y - y_pred)

def gradient_descent(X, y, config):
    loss_his = []
    w = np.random.randint(100, size=1)[0]
    b = np.random.randint(100, size=1)[0]
    X_dump = np.arange(config["len_dataset"] + 1)

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 5))    
    plt.title("Traning Process")
    plt.scatter(X, y)

    line1, = ax.plot(X_dump, inference(X_dump, w, b), color='green')


    plt.xlabel("Area")
    plt.ylabel("Price")
    for _ in range(config['epochs']):
        pred = inference(X, w, b)
        loss = MSELoss(y, pred)
        loss_his.append(loss)

        w = w - config["learning_rate"] * w_grad(X, y, pred)
        b = b - config["learning_rate"] * b_grad(y, pred)

        y_dump = inference(X_dump, w, b)

        line1.set_xdata(X_dump)
        line1.set_ydata(y_dump)
        
        figure.canvas.draw()
        figure.canvas.flush_events()
 
        time.sleep(0.5)
    
    plt.show()

    return w, b, loss_his

if __name__ == '__main__':
    config = {
        'len_dataset': 200,
        'learning_rate': 0.000001,
        'epochs': 60,
    }

    X, y = create_dummy_dataset(config["len_dataset"])
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y)
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Data Visualization")
    plt.show()

    w, b, loss_his = gradient_descent(X, y, config)