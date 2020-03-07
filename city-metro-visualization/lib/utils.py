#!/usr/bin/env python
#coding=utf-8
# @file  : utils
# @time  : 3/7/2020 2:55 PM
# @author: shishishu

import os

def safe_mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    return