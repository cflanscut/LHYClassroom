#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2020/12/04 09:45:03
@Author      :chaofan
@version      :1.0
'''
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def readfile(path, label_needed=False):
    """
    @description  :tranform the raw data into vectors
    ---------
    @param  :path is the input system path, label_needed means that the inputs
     and outputs have labels
    -------
    @Returns  :x(return picture vector),y(return picture's label if neccessary)
    -------
    """
    image_dir = sorted(
        os.listdir(path))  # 输入path，范围该path下的dir，即所有文件名，并用sorted排序
    x = np.zeros((len(image_dir), 128, 128, 3),
                 dtype=np.uint8)  # 根据path下面的长度构造容器长度
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))  # 用相对路径访问文件
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label_needed:
            y[i] = int(file.split("_")[0])
    if label_needed:
        return x, y
    else:
        return x


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),  # 把图片转成tensor，并normalization
])  # Compose是一个类，存储了变换的指令和函数
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
