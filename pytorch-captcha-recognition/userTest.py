#!/usr/bin/env python
#coding=utf-8

from model import *
from dataset import *
from train import *
import time


def userTest(model, dataLoader, convert_lower=False):
    totalNum = 0
    rightNum = 0
    for circle, input in enumerate(dataLoader, 0):
        if circle >= 200:
            break
        totalNum += 1
        x, label = input
        if t.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        realLabel = LabeltoStr([label[0][0], label[0][1], label[0][2], label[0][3]])
        y1, y2, y3, y4 = model(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        if convert_lower:
            realLabel = realLabel.lower()  # added (no difference between lower and upper)
            decLabel = decLabel.lower()  # added (no difference between lower and upper)
        # print("real: %s -> %s , %s" % (realLabel, decLabel, str(realLabel == decLabel)))
        if realLabel == decLabel:
            rightNum += 1
    print("\n total %s, right %s" % (totalNum, rightNum))

if __name__ == '__main__':
    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist()
    if t.cuda.is_available():
        model = model.cuda()
    userTestDataset = Captcha("./golden_set/", train=False)
    userTestDataLoader = DataLoader(userTestDataset, batch_size=1, shuffle=False, num_workers=0)
    start = time.time()
    userTest(model, userTestDataLoader, True)
    print('time cost is: ', time.time()-start)