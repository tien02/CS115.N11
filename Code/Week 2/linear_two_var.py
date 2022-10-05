import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_dataset():
    x1 = np.expand_dims(np.arange(10, 160), axis=0)
    x2 = np.expand_dims(np.arange(5, 155), axis=0)
    X = np.concatenate((x1, x2), axis=0)
    w = np.array([[2], [3]])
    b = 7
    y = np.dot(X.T, w) + b + np.random.randint(-1, 2, size=(X.shape[1], 1))
    return X, y

def MSELoss(y, y_pred):
    return np.mean(np.square(y - y_pred))

def inference(x, w, b):
    return np.dot(x.T, w) + b

def w_grad(x, y, y_pred):
    return -2 * np.mean(np.dot(x, y-y_pred))

def b_grad(y, y_pred):
    return -2 * np.mean(y - y_pred)

def train_test_split(X, y, train_size = 100):
    X_train = X[:, :train_size]
    y_train = y[:train_size, :]
    X_test = X[:, train_size:]
    y_test = y[train_size:, :]
    return X_train, X_test, y_train, y_test

def gradient_descent(X, y, config):
    loss_his = []
    # w = np.random.randint(1, 4, size=(2,1))
    w1 = np.random.uniform(1.9, 2.1)
    w2 = np.random.uniform(2.9, 3.1)
    w = np.array([[w1], [w2]])
    b = np.random.uniform(6.9, 7.1)

    for idx in tqdm(range(config['epochs'])):
        pred = inference(X, w, b)
        loss = MSELoss(y, pred)
        loss_his.append(loss)

        w = w - config["learning_rate"] * w_grad(X, y, pred)
        b = b - config["learning_rate"] * b_grad(y, pred)

    return w, b, loss_his

if __name__ == '__main__':
    config = {
        'learning_rate': 0.0000001,
        'epochs': 15000,
    }

    print("\n\t** LOAD DATA **\n")
    X, y = create_dataset()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    plt.figure(figsize=(10, 5))
    plt.scatter(X[0, :], y)
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Data Visualization")
    plt.show()
    print("-" * 69)

    print("\n\t** SPLIT DATA **\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("-" * 69)

    print("\n\t** TRAINING **\n")
    w, b, loss_his = gradient_descent(X_train, y_train, config)

    plt.plot(loss_his)
    plt.title("Loss over time")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.show()
    print(f"\nW: {w}")
    print(f"b: {b}")
    print("-" * 69)

    print("\n\t** LOSS **\n")
    print(f"MSE loss on TRAIN set: {loss_his[-1]:.3f}")
    print(f"MSE loss on TEST set: {MSELoss(y_test, inference(X_test, w, b)):.3f}")
    print("-" * 69)