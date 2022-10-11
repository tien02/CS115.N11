import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def BMIDataset(num_sample=1000):
    weight = np.random.randint(30, 120, size=num_sample)
    height = np.random.randint(130, 200, size=num_sample) / 100
    data = np.concatenate(([weight], [height]), axis=0)
    data = data.T
    bmi = weight / np.square(height)
    bmi = bmi >= 23
    bmi = bmi.astype(int)
    bmi = bmi.T
    return data, np.expand_dims(bmi, axis=1)

def BCELoss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1-y_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inference(x, w, b):
    return sigmoid(np.dot(x, w) + b)

def gradient(x, y, y_pred):
    m = x.shape[0]

    dw = np.dot(x.T, y_pred-y) / m
    db = np.sum(y_pred-y) / m

    return dw, db

def gradient_descent(X, y, lr, epochs):
    loss_his = []
    w = np.random.rand(2,1)
    b = np.random.rand(1)[0]

    for _ in tqdm(range(epochs)):
        pred = inference(X, w, b)
        loss = BCELoss(y, pred)
        loss_his.append(loss)

        dw, db = gradient(X, y, pred)
        w = w - lr * dw
        b = b - lr * db
    
    return w, b, loss_his

def predict(x, w, b, threshold=0.5):
    y_pred = inference(x, w, b)
    y_pred = y_pred >= threshold
    y_pred = y_pred.astype('int')
    return y_pred

def accuracy(y, y_pred):
    acc = y == y_pred
    acc = np.array(acc, dtype="float")
    return np.mean(acc)

def normalize(X):
    X_norm = X / np.max(X, axis=0)
    return X_norm

def train_test_split(X, y, test_size=0.2):
    data = np.concatenate((X, y), axis=1)
    train_size = int(data.shape[0] * (1 - test_size))
    X_train, y_train = data[:train_size, :-1], np.expand_dims(data[:train_size, -1], axis=1)
    X_test, y_test = data[train_size:, :-1], np.expand_dims(data[train_size:, -1], axis=1)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = BMIDataset(1000)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    plt.title("Data")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Height (m)")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    np.random.seed(60)
    X_train_norm = normalize(X_train)
    w, b, loss_his = gradient_descent(X_train_norm, y_train, 0.01, 1000)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss_his)
    plt.show()

    X_test_norm = normalize(X_test)
    y_test_pred = predict(X_test_norm, w, b)
    print(f"Accuracy on test set: {accuracy(y_test, y_test_pred):.2f}")