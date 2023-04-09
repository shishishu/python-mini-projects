#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 5 May 2019
@author: Nano Zhou
- ref: https://github.com/braveryCHR/CNN_captcha
"""

from parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=charLength):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [3, 32, 32] -> [64, 32, 32]
        x = self.layer1(x)  # -> [64, 32, 32] -> [64, 32, 32]
        x = self.layer2(x)  # -> [128, 16, 16] -> [128, 16, 16], floor in conv2d
        x = self.layer3(x)  # -> [256, 8, 8] -> [256, 8, 8]
        x = self.layer4(x)  # -> [512, 4, 4] -> [512, 4, 4]
        x = F.avg_pool2d(x, 4)  # -> [512, 1, 1]
        x = x.view(-1, 512)  # -> [1, 512]
        y1 = self.fc1(x)  # -> [1, 62]
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./model/resNet" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = "./model/resNet_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self):
        fileList = os.listdir("./model/")
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = "./model/resNet_new.pth"
            if t.cuda.is_available():
                self.load_state_dict(t.load(name))
            else:
                self.load_state_dict(t.load(name, map_location='cpu'))  # load model (trained on gpu) in PC with cpu-only
            print("the latest model has been load")