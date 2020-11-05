import numpy as np


def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def function(x, w, b):
    return sigmoid(np.matmul(x, w) + b)


def predict(x, w, b):
    return np.round(function(x, w, b)).astype(np.int)


def reshuffle(x, y):
    np.random.seed(0)
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return (x[randomize], y[randomize])


def accurancy(y_predicition, y_label):
    return 1 - np.mean(np.abs(y_predicition - y_label))


def crossentropy_loss(y_predicition, y_label):
    crossentropy = -np.dot(y_label, np.log(y_predicition)) - np.dot(
        1 - y_label, np.log(1 - y_predicition))
    return crossentropy


def gradient(x, y_label, w, b):
    y_predicition = function(x, w, b)
    w_gradient = -np.sum(
        (y_label - y_predicition) * x.T + b,
        1)  # 这里用的是*号，即y是1*N的相量，与x.T(k*N维度)的每一行相乘，然后得到k*N的结果，再对每一列求和
    b_gradient = -np.sum((y_label - y_predicition) * x.T)
    return w_gradient, b_gradient
