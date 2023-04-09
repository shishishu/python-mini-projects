#!/usr/bin/env python
#coding=utf-8

from parameters import *
from model import *
from train import *

if __name__ == '__main__':
    net = ResNet(ResidualBlock)
    net.loadIfExist()
    '''
    for elem in net.named_parameters():
        print(elem)
    '''
    train(net)