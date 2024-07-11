#!/usr/bin/env python
#coding=utf-8
# @file  : config
# @time  : 2/13/2022 10:42 PM
# @author: shishishu

import os

current_path = os.path.abspath(__file__)
master_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + '..')

data_dir = os.path.join(master_path, 'data')
data_dir_linux = '/mnt/d/app-museum/data'
archive_dir = data_dir.replace('D', 'E')
archive_dir_linux = '/mnt/e/app-museum/data'


def linux2windows_data_path(linux_data_path):
    relative_path = linux_data_path.split('data/')[1].split('/')
    relative_path = '\\'.join(relative_path)
    return os.path.join(archive_dir, relative_path)

# a = '/mnt/e/app-museum/data/SHM/0/0/img/CI00004860_o.jpg'
# b = linux2windows_data_path(a)
# print(a, b, os.path.exists(b))
