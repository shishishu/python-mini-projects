#!/usr/bin/env python
#coding=utf-8
# @file  : config
# @time  : 2/23/2020 4:02 PM
# @author: shishishu

import os
from collections import OrderedDict

current_path = os.path.abspath(__file__)
master_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + '..')

data_dir = master_path + './data'
dict_dir = master_path + './dict'

pos_map = OrderedDict([
    ('n', '名词'),
    ('v', '动词'),
    ('a', '形容词'),
    ('d', '副词'),
    ('z', '状态词'),
    ('nr', '人名/物名'),
    ('vn', '名动词'),
    ('l', '习惯用语'),
    ('i', '成语'),
    ('m', '数词'),
    ('ns', '地名')
])