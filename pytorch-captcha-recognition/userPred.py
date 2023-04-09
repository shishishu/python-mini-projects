#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 5 May 2019
@author: Nano Zhou
"""

from model import *
from train import *
from parameters import *

from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as  T

import time
import glob


class Captcha2(data.Dataset):

    def __init__(self, imgsPath, train=True):
        self.imgsPath = [imgsPath]  # only one image passed at once
        self.transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        data = Image.open(imgPath)
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.imgsPath)

def userPred(model, dataLoader):
    x = list(dataLoader)[0]  # [1, 3, 32, 32]
    y1, y2, y3, y4 = model(x)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = torch.cat((y1, y2, y3, y4), dim=1)
    decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
    # decLabel = decLabel.lower()  # added (no difference between lower and upper)
    return decLabel

def predict_image(model, imgPath):
    x = Image.open(imgPath)
    transform = T.Compose([
        T.Resize((ImageHeight, ImageWidth)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    x = transform(x).unsqueeze(0)  # [1, 3, 32, 32]
    y1, y2, y3, y4 = model(x)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = torch.cat((y1, y2, y3, y4), dim=1)
    decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
    # decLabel = decLabel.lower()  # added (no difference between lower and upper)
    return decLabel


if __name__ == '__main__':

    start = time.time()
    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist()
    print('time cost in loading model is: ', time.time()-start)

    print('target captcha is: C4j2')
    
    start = time.time()
    userPredDataset = Captcha2('./golden_set/C4j2.jpg', train=False)
    userPredDataLoader = DataLoader(userPredDataset, batch_size=1, shuffle=False, num_workers=0)  # it will be much slower if num_workers > 0 
    pred = userPred(model, userPredDataLoader)
    print('predition is: ', pred)
    print('time cost in prediction with dataLoader is: ', time.time()-start)
    
    start = time.time()
    pred = predict_image(model, './golden_set/C4j2.jpg')
    print('predition is: ', pred)
    print('time cost in prediciton without dataLoader is: ', time.time()-start)  # recommend in runtime service