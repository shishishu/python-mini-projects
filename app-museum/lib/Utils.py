#!/usr/bin/env python
#coding=utf-8
# @file  : utils
# @time  : 2/13/2022 6:37 PM
# @author: shishishu

import pypinyin
import requests
import os

class Utils:

    # 不带声调的(style=pypinyin.NORMAL)
    @staticmethod
    def char2pinyin(word):
        s = ''
        for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
            s += ''.join(i)
        return s.upper()

    @staticmethod
    def download_image(img_url, img_path):
        res = requests.get(img_url)
        with open(img_path, 'wb') as fw:
            fw.write(res.content)
        return

