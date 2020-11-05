import numpy as np
import math
import csv

with open('./task2/X_test', 'r') as f:
    next(f)
    x_test = np.array([line.strip('\n').split(',')[1:] for line in f],
                      dtype=float)
with open('./task2/X_train', 'r') as f:
    next(f)
    x_train = np.array([line.strip('\n').split(',')[1:] for line in f],
                       dtype=float)
with open('./task2/Y_train', 'r') as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1:] for line in f],
                       dtype=float)


def normalise(x,
              is_training_data=False,
              specified_column=None,
              x_mean=None,
              x_std=None):
    if specified_column is None:
        specified_column = np.arange(x.shape[1])
    if is_training_data:
        x_mean = np.mean(x[:, specified_column], 0)
        x_std = np.std(x[:, specified_column], 0)
    x[:,
      specified_column] = (x[:, specified_column] - x_mean) / (x_std + 1e-10)
    return x, x_mean, x_std


def train_data_split(x, y, split_ratio=0.25):
    training_size = math.floor(len(x) * split_ratio)
    return x[:training_size, :], y[:training_size, :], x[training_size:, :], y[
        training_size:, :]


x_train, x_mean, x_std = normalise(x_train, is_training_data=True)
x_test, _, _ = normalise(x_test,
                         is_training_data=False,
                         specified_column=None,
                         x_mean=x_mean,
                         x_std=x_std)
with open('./task2/train_normalised.csv', mode='w', newline='') as file:
    file_writer = csv.writer(file)
    for i in x_train:
        file_writer.writerow(i)
with open('./task2/test_normalised.csv', mode='w', newline='') as file:
    file_writer = csv.writer(file)
    for i in x_test:
        file_writer.writerow(i)
