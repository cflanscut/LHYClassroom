import numpy as np


def Adagrad(x, y):
    """
    :param x:训练样本
    :param y:训练标签
    """
    try:
        dim = len(x[0]) + 1
        w = np.zeros([dim, 1], dtype=float)
        adagrad = np.zeros([dim, 1], dtype=float)
        x = np.concatenate((np.ones([len(x), 1]), x), axis=1)
        x_t = x.transpose()
        learning_rate = 100
        iter_time = 10000
        eps = 0.0000000001
        for i in range(iter_time):
            loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
            if (i % 1000 == 0):
                print(str(i) + ":" + str(loss))
            temp = np.dot(x, w) - y
            gradient = 2 * np.dot(x_t, temp)
            adagrad += gradient**2
            w = w - learning_rate * gradient / (np.sqrt(adagrad + eps))
    except ValueError as err:
        print(err)
    return w
