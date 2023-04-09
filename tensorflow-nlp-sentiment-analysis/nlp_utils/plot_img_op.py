#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 20 Apr 2019
@author: Nano Zhou
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def export_image(output_dir, epoch, y_true, y_pred):
    y_true_stat = Counter(y_true)
    y_pred_stat = Counter(y_pred)  # dict but un-ordered
    # keys = sorted(y_true_stat.keys()) # all the sorted keys, e.g.[0,1,2,3]
    keys = [0, 1, 2, 3]
    y_true_list = [y_true_stat[i] for i in keys]
    y_pred_list = [y_pred_stat[i] for i in keys]
    x = np.arange(len(keys))
    width = 0.4
    _, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, y_true_list, width, color='b', label='true')
    ax.bar(x+width, y_pred_list, width, color='y', label='pred')
    ax.set_xticks(x+width/2)
    ax.set_xticklabels([-2, -1, 0, 1])
    for a, b in zip(x, y_true_list):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
    for a, b in zip(x, y_pred_list):
        plt.text(a+width, b, b, ha='center', va='bottom', fontsize=12)
    plt.xlabel('class type', fontsize=15)
    plt.ylabel('count of each class', fontsize=15)
    plt.title('prediction at epoch {}'.format(epoch), fontsize=20)
    plt.legend()
    plt.savefig(output_dir + '/epoch_' + str(epoch) + '.png')