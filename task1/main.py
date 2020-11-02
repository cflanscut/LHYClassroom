from Adagrad import Adagrad
import pandas as pd
import numpy as np

# training
data = pd.read_csv('./task1/training_data.csv', header=None)
data = data.to_numpy()
x = data[:, :len(data[0]) - 1]
y = np.empty([len(data), 1])
y[:, 0] = data[:, len(data[0]) - 1]
w = Adagrad(x, y)

# validating
vali_data = pd.read_csv('./task1/validation_data.csv', header=None)
vali_data = vali_data.to_numpy()
vali_x = vali_data[:, :len(vali_data[0]) - 1]
vali_x = np.concatenate((np.ones([len(vali_x), 1]), vali_x), axis=1)
vali_y = np.empty([len(vali_data), 1])
vali_y[:, 0] = vali_data[:, len(vali_data[0]) - 1]
predict_y = np.dot(vali_x, w)
loss = np.sqrt(np.sum(np.power(predict_y - vali_y, 2)) / len(vali_data))
print('validation loss:' + str(loss))
np.save('weight.npy', w)
