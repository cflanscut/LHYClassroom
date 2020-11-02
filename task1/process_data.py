import pandas as pd
import numpy as np
import math
import csv

# processing data
training_file = pd.read_csv(r"./task1/train.csv", encoding='big5')
training_set = training_file.iloc[:, 3:]
training_set[training_set == 'NR'] = 0
raw_data = training_set.to_numpy()

# extract feature
mouth_data = {}
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for mouth in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24:(day + 1) * 24] = raw_data[(mouth * 20 + day) *
                                                      18:(mouth * 20 +
                                                          (day + 1)) * 18, :]
    sample_T = sample.transpose()
    for i in range(471):
        for j in range(9):
            x[mouth * 471 + i, j * 18:(j + 1) * 18] = sample_T[i + j, :]
        y[mouth * 471 + i, 0] = sample_T[i + 9, 9]

# normalize data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if x_std[j] != 0:
            x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]

# seperate data
x_training_data = x[:math.floor(len(x) * 0.8), :]
y_training_data = y[:math.floor(len(y) * 0.8), :]
x_validation_data = x[math.floor(len(x) * 0.8):, :]
y_validation_data = y[math.floor(len(y) * 0.8):, :]
training_data = np.concatenate((x_training_data, y_training_data), axis=1)
validation_data = np.concatenate((x_validation_data, y_validation_data),
                                 axis=1)
with open(r'./task1/training_data.csv', mode='w',
          newline='') as training_data_file:
    csv_writer = csv.writer(training_data_file)
    for i in range(len(training_data)):
        row = training_data[i]
        csv_writer.writerow(row)
with open(r'./task1/validation_data.csv', mode='w',
          newline='') as validation_data_file:
    csv_writer = csv.writer(validation_data_file)
    for i in range(len(validation_data)):
        row = validation_data[i]
        csv_writer.writerow(row)
