# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

df = pd.read_csv('Iris.csv')
df.head()

# normalized between (0, 1) to prevent gradient vanishing caused by out sigmoid
X = MinMaxScaler().fit_transform(df.iloc[:,1:-1].values)
Y = LabelEncoder().fit_transform(df.iloc[:, -1])
onehot = np.zeros((Y.shape[0], np.unique(Y).shape[0]))
onehot[range(Y.shape[0]),Y] = 1.0
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,1:-1].values, onehot,test_size=0.2)

def sigmoid(X, grad=False):
    if grad:
        return sigmoid(X) * (1 - sigmoid(X))
    else:
        return 1 / (1 + np.exp(-X))
    
def softmax(X, grad=False):
    if grad:
        p = softmax(X)
        return p * (1-p)
    else:
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy(X, Y, grad=False):
    if grad:
        X = np.clip(X, 1e-15, 1 - 1e-15)
        return -(Y / X) + (1 - Y) / (1 - X)
    else:
        X = np.clip(X, 1e-15, 1 - 1e-15)
        return -Y * np.log(X) - (1 - Y) * np.log(1 - X)

W1 = np.random.randn(X_train.shape[1], 50) / np.sqrt(df.shape[1])
b1 = np.zeros((50))
W2 = np.random.randn(50, 100) / np.sqrt(50)
b2 = np.zeros((100))
W3 = np.random.randn(100, np.unique(Y).shape[0]) / np.sqrt(100)
b3 = np.zeros((np.unique(Y).shape[0]))

EPOCH = 50
LEARNING_RATE = 0.01

# a(x, w) = x.dot(w)
# z(x) = sigmoid(x)
# y_hat(x) = softmax(x)
# E(y_hat, y) = cross_entropy(y_hat, y)

# feed-forward
# a1(x, w1) -> z1(a1) -> a2(z1, w2) -> z2(a2) -> a3(z2, w3) -> y_hat(a3) -> E(y_hat, y)

# back-propagation
# a1(x, w1) <- dz1(a1) <- da2(z1, w2) <- dz2(a2) <- da3(z2, w3) <- dy_hat(a3) <- dE(y_hat, y)

for i in range(EPOCH):
    a1 = np.dot(X_train, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y_hat = softmax(a3)
    accuracy = np.mean(np.argmax(y_hat,axis = 1) == np.argmax(Y_train,axis = 1))
    cost = np.mean(cross_entropy(y_hat,Y_train))
    dy_hat = cross_entropy(y_hat,Y_train, grad=True)
    da3 = softmax(a3, True) * dy_hat
    dW3 = z2.T.dot(da3)
    db3 = np.sum(da3, axis=0)
    dz2 = da3.dot(W3.T)
    da2 = sigmoid(a2, True) * dz2
    dW2 = z1.T.dot(da2)
    db2 = np.sum(da2, axis=0)
    dz1 = da2.dot(W2.T)
    da1 = sigmoid(a1, True) * dz1
    dW1 = X_train.T.dot(da1)
    db1 = np.sum(da1, axis=0)
    W3 += -LEARNING_RATE * dW3
    b3 += -LEARNING_RATE * db3
    W2 += -LEARNING_RATE * dW2
    b2 += -LEARNING_RATE * db2
    W1 += -LEARNING_RATE * dW1
    b1 += -LEARNING_RATE * db1
    print('epoch %d, accuracy %f, cost %f'%(i, accuracy, cost))

