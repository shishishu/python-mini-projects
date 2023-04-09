#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue, 30 Apr 2019
@author: Nano Zhou
"""

import numpy as np
import cv2
import pytesseract
import base64
import glob
import os
import multiprocessing


def get_dir_file_name(file_path):
    dir_name, file_name = os.path.split(file_path)
    file_name_stem, file_name_suffix = file_name.split('.')
    return dir_name, file_name_stem, file_name_suffix

def convert_base64_to_img(file_path, dir_name='./input/google_ocr/'):
    _, file_name_stem, _ = get_dir_file_name(file_path)
    with open(file_path, 'r') as fr:
        base64_data = fr.read().split(',')[1]
        img_data = base64.b64decode(base64_data)
        with open(dir_name + file_name_stem + '.jpg', 'wb') as fw:
            fw.write(img_data)

def remove_line(im_input, deltaX=20):
    im = im_input.copy()
    height, width = im.shape
    
    for y in range(width):
        for x in range(6, height-6):
            deltaX_u = im[x, y] - im[x-1, y]
            deltaX_d = im[x, y] - im[x+1, y]
            if deltaX_u * deltaX_d > 0:
                if abs(deltaX_u) > deltaX and abs(deltaX_d) > deltaX:
                    for i in range(x-3, x):
                        im[i, y] = im[x-4, y]
                    for i in range(x+1, x+4):
                        im[i, y] = im[x+4, y]
                    im[x, y] = int((im[x-1, y] + im[x+1, y]) / 2)
    return im

def img_process(img_path, dir_name='./input/google_ocr/', split_number=4):
    _, file_name_stem, _ = get_dir_file_name(img_path)

    # im = cv2.imread(img_path)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.imread(img_path, 0)  # load as gray img directly

    im_32 = np.array(im, dtype=np.int32)
    im_32 = remove_line(im_32, 20)
    im = np.array(im_32, dtype=np.uint8)

    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 1)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    length = opening.shape[1]
    split_length = int(length/split_number)
    assert split_length == 16, 'wrong split'
    
    config = ('--oem 0 --psm 10 Nano')  # revise whitelist and save as 'Nano', output char or number only
    # 'Nano' is located in C:\Program Files (x86)\Tesseract-OCR\tessdata\configs, copy 'digits' and revise whitelist

    for i in range(split_number):
        sub_im = opening[:, split_length*i:split_length*(i+1)]
        pred = pytesseract.image_to_string(sub_im, config=config)
        sub_im_path = dir_name + file_name_stem + '_' + str(i) + '_' + str(pred) + '.jpg'
        cv2.imwrite(sub_im_path, sub_im)

def pool_initializer():
    global jpg_file_names
    jpg_file_names = set(glob.glob('./input/google_ocr/*.jpg'))  # use set as it makes search faster than list

def img_pipeline(file_path):
    _, file_name_stem, _ = get_dir_file_name(file_path)
    img_path = './input/google_ocr/' + file_name_stem + '.jpg'
    if img_path in jpg_file_names:
        return
    convert_base64_to_img(file_path)
    img_process(img_path)


if __name__ == '__main__':
    
    # convert_base64_to_img('./input/origin_base64/1.txt')
    # img_process('./input/google_ocr/1.jpg')
    
    pool = multiprocessing.Pool(processes=4, initializer=pool_initializer)
    file_names = glob.glob('./input/origin_base64/*.txt')
    pool.map(img_pipeline, [file_path for file_path in file_names])