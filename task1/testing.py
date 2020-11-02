import pandas as pd
import numpy as np
import math
w = np.load('weight.npy')
x_mean = np.load('x_mean.npy')
x_std = np.load('x_std.npy')
testing_file = pd.read_csv('./task1/test.csv', header=None)
testing_set = testing_file.iloc[0:, 2:]
testing_set[testing_set == 'NR'] = 0
testing_set = testing_set.to_numpy()
x = np.empty([math.floor(len(testing_set) / 18), 162], dtype=float)
testing_set_t = testing_set.transpose()
for i in range(len(x)):
    for j in range(9):
        x[i, j * 18:(j + 1) * 18] = testing_set_t[j, i * 18:(i + 1) * 18]
    for j in range(len(x[0])):
        if x_std[j] != 0:
            x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]
x = np.concatenate((np.ones([len(x), 1]), x), axis=1)
y = np.dot(x, w)
for row in range(len(y)):
    y[row] = math.floor(y[row])
    if y[row] <= 0:
        y[row] = 1
    print(str(y[row]))
