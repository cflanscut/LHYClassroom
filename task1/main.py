import pandas as pd
import numpy as np

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
