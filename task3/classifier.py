#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2020/12/08 15:52:19
@Author      :chaofan
@version      :1.0
'''
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 128x128x3经过核长3，64个核的卷积，得到128x128x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 经过2长的池化，直接除2得到64x64x64
            nn.Conv2d(64, 128, 3, 1, 1),  # 64x64x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 32x32x128
            nn.Conv2d(128, 256, 3, 1, 1),  # 32x32x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 16x16x256
            nn.Conv2d(256, 512, 3, 1, 1),  # 16x16x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 8x8x512
            nn.Conv2d(512, 512, 3, 1, 1),  # 8x8x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 4x4x512 即
        )
        self.fc = nn.Sequential(nn.Linear(512 * 4 * 4, 1024), nn.ReLU(),
                                nn.Linear(1024, 512), nn.ReLU(),
                                nn.Linear(512, 11))

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
