#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2020/12/08 16:51:50
@Author      :chaofan
@version      :1.0
'''
# import sys
# if "/task3/" not in sys.path:
#     sys.path.append('/task3/')
# if 'classifier' not in sys.modules:
#     classifier = __import__('classifier')
# else:
#     eval('import classifier')
#     classifier = eval('reload(classifier)')
# from classifier import Classifier

from image_ready import readfile, ImgDataset, test_transform, train_transform
from classifier import Classifier
import torch.nn as nn
import torch
import time
import numpy as np
import os
from torch.utils.data import DataLoader
import sys
sys.path.append(r'./task3')

torch.cuda.set_device(2)

# 读取数据
workspace_dir = './task3/food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, 'training'), True)
print("Size of training data={}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print("Size of validation data={}".format(len(val_x)))

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set,
                              batch_size=batch_size,
                              shuffle=True)

# 建立模型
model = Classifier().cuda()  # 初始化model
loss = nn.CrossEntropyLoss()  # cep也是一个类，初始化
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30

# 训练模型
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 这个是干嘛？设置为训练模式，在dropout和batchNorm才生效，与eval对应
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        # 前馈,得到预测 data[0]即x，用.cuda应该是把数据丢入cuda？
        batch_loss = loss(train_pred, data[1].cuda())  # data[1]即y；
        batch_loss.backward()
        # 这一步是计算gradient。loss()是计算新的标量的函数，如果传入的变量是grad=True，那么就会在各自参数保存梯度
        optimizer.step()
        # 优化器怎么知道用哪个gradient更新参数？gradient已经存在parameters类里面 param.grad
        # 代替手动param=param-learning_rate*grad,用adam的方式，参考onenote里adagrad那里
        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
            data[1].numpy())
        # argmax返回最大值的索引，前面是用独热编码，所以哪个位置最大，就是哪一类
        # 这里用比较是什么意思?
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        # 可以禁用梯度计算，减少内存，加快速度
        # with后面的表达式必须要有enter和exit函数，类似于异常处理
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(
                np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            val_loss += batch_loss.item()

# 接下来用全部数据进行训练
model_best = Classifier().cuda()  # 初始化model
loss = nn.CrossEntropyLoss()  # cep也是一个类，初始化
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
num_epoch = 30
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model_best.train()  # 这个是干嘛？设置为训练模式，在dropout和batchNorm才生效，与eval对应
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        # 前馈,得到预测 data[0]即x，用.cuda应该是把数据丢入cuda？
        batch_loss = loss(train_pred, data[1].cuda())  # data[1]即y；
        batch_loss.backward()
        # 这一步是计算gradient。loss()是计算新的标量的函数，如果传入的变量是grad=True，那么就会在各自参数保存梯度
        optimizer.step()
        # 优化器怎么知道用哪个gradient更新参数？gradient已经存在parameters类里面 param.grad
        # 代替手动param=param-learning_rate*grad,用adam的方式，参考onenote里adagrad那里
        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
            data[1].numpy())
        # argmax返回最大值的索引，前面是用独热编码，所以哪个位置最大，就是哪一类
        # 这里用比较是什么意思?
        train_loss += batch_loss.item()

    model_best.eval()
    with torch.no_grad():
        # 可以禁用梯度计算，减少内存，加快速度
        # with后面的表达式必须要有enter和exit函数，类似于异常处理
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(
                np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            val_loss += batch_loss.item()

# 保存模型参数
torch.save(model.state_dict(), './task3/model.pth')
torch.save(model_best.state_dict(), './task3/model_best.pth')
print('训练成功')