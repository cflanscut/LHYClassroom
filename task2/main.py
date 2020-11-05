import logistic_regression as lgr
import pandas as pd
import numpy as np

training_data = pd.read_csv('./task2/train_normalised.csv', header=None)
training_data = training_data.to_numpy()
testing_data = pd.read_csv('./task2/test_normalised.csv', header=None)
testing_data = testing_data.to_numpy()
label = np.empty([len(training_data), 1])
with open('./task2/Y_train', mode='r') as f:
    next(f)
    training_label = np.array([line.strip('\n').split(',')[1:] for line in f],
                              dtype=float)
    for i in range(len(training_label)):
        label[i, :] = training_label[i]
w = np.zeros((training_data.shape[1], ), dtype=float)
b = np.zeros((1, ), dtype=float)

# iterate
learning_rate = 0.002
iter_time = 100
batch_size = 8
batch_lenth = np.int(np.floor(len(training_data) / batch_size))

train_acc = []
test_acc = []
train_loss = []
test_acc = []
step = 1
for i in range(iter_time):
    training_data, label = lgr.reshuffle(training_data, label)
    for j in range(batch_size):
        X = training_data[j * batch_lenth:(j + 1) * batch_lenth, :]
        Y = label[j * batch_lenth:(j + 1) * batch_lenth, :]
        w_grad, b_grad = lgr.gradient(X, Y, w, b)
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad
        step += 1
    y_train_predict = lgr.function(training_data, w, b)
    Y_train_predict = np.round(y_train_predict)
    train_accurancy = lgr.accurancy(Y_train_predict, label)
    train_crossentropy = lgr.crossentropy_loss(y_train_predict, label)
    train_acc.append(train_accurancy)
    train_loss.append(train_crossentropy / len(training_data))
print('training accurancy:{}'.format(train_acc[-1]))
print('training loss:{}'.format(train_loss[-1]))
