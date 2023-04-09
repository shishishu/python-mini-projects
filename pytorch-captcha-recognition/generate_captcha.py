#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 5 May 2019
@author: Nano Zhou
"""

import random
import string
from custom_captcha_image import *
from parameters import *


def get_string():
    source = []
    for _ in range(charNumber):
        select = random.randint(1, 3)
        if select == 1:
            source.extend(random.sample(string.digits, 1))
        elif select == 2:
            source.extend(random.sample(string.ascii_lowercase, 1))
        else:
            source.extend(random.sample(string.ascii_uppercase, 1))
    return ''.join(source)

def get_captcha(num, path):
    for i in range(num):
        if i > 0 and i % 10000 == 0: 
            print(i)
        imc = ImageCaptcha(ImageWidth_custom, ImageHeight_custom, font_sizes=font_sizes_custom)
        name = get_string()
        file_name = path + name + '.jpg'
        try:
            imc.write(name, file_name)  # inherit from _Captcha class
        except FileNotFoundError:
            print('File {} not found'.format(file_name))


if __name__ == '__main__':
    get_captcha(10, "./data/userTest/")