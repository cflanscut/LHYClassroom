#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2020/12/14 19:56:44
@Author      :chaofan
@version      :1.0
'''

from image_ready import readfile, ImgDataset, test_transform
from classifier import Classifier
import torch
import os
from torch.utils.data import DataLoader
import sys
import numpy as np
sys.path.append(r'/home/chaofan/LHYCLASSROOM/task3')

# 先把训练好的模型读进来
model_best_state_dict = torch.load('./task3/model.pth')
model_best = Classifier().cuda()
model_best.load_state_dict(model_best_state_dict)

# 读取训练所要用的数据
workspace_dir = './task3/food-11'
test_x = readfile(os.path.join(workspace_dir, 'testing'))
print("Size of testing data={}".format(len(test_x)))
batch_size = 128
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 测试数据
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

print(prediction)