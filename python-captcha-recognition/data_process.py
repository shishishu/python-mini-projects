#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 3 May 2019
@author: Nano Zhou
"""

import glob
import json
import cv2
import pickle
import numpy as np
from img_google_ocr import get_dir_file_name

var_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',\
    'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def decode_jpg_name(img_path, key_index):
    _, img_name_stem, _ = get_dir_file_name(img_path)
    name_split = img_name_stem.split('_')
    if len(name_split) == 3 and name_split[0] == str(key_index) and name_split[-1] != '':
        return img_path, name_split[-1].lower()
    return False

def gen_jpg_dict(output_path, max_num=99999, dir_name='./input/google_ocr/'):
    jpg_dict = dict()
    img_names = glob.glob(dir_name + '/*.jpg')
    # sorted files: 1.jpg, 1_0_a.jpg, 1_1_b.jpg, 1_2_c.jpg, 1_3_d.jpg, 2.jpg...
    sorted_img_names = sorted(img_names, key=lambda x: int(get_dir_file_name(x)[1].split('_')[0]))  # sorted by file name
    for i in range(1, max_num + 1):
        tmp_list = []
        for j in range(1, 5):
            tmp_img_path = sorted_img_names[5 * (i-1) + j]
            decode_result = decode_jpg_name(tmp_img_path, i)
            if decode_result:
                tmp_list.append(decode_result)
        jpg_dict[str(i)] = tmp_list
    with open(output_path, 'w') as fw:
        json.dump(jpg_dict, fw)
    return jpg_dict

def gen_confirm_dict(output_path, max_num=99999, dir_name='./input/baidu_api/'):
    word_confirm_dict = dict()
    json_names = glob.glob(dir_name + '/*.json')
    sorted_json_names = sorted(json_names, key=lambda x: int(get_dir_file_name(x)[1]))
    for i in range(1, max_num + 1):
        tmp_json_name = sorted_json_names[i-1]
        _, json_name_stem, _ = get_dir_file_name(tmp_json_name)
        assert json_name_stem == str(i), 'wrong matching...'
        with open(tmp_json_name, 'r') as fr:
            word_result = json.load(fr)['words_result']
            if word_result != []:
                word_confirm_dict[str(i)] = word_result[0]['words'].lower()
            else:
                word_confirm_dict[str(i)] = []
    with open(output_path, 'w') as fw:
        json.dump(word_confirm_dict, fw)
    return word_confirm_dict

def gen_correct_label_dict(google_jpg_dict, baidu_confirm_dict, output_path, max_num=99999):
    correct_label_dict = dict()
    for i in range(1, max_num + 1):
        jpg_item = google_jpg_dict[str(i)]
        confirm_item = baidu_confirm_dict[str(i)]
        match_result = [elem for elem in jpg_item if elem[1] in confirm_item]
        if match_result != []:
            correct_label_dict[str(i)] = match_result
    with open(output_path, 'w') as fw:
        json.dump(correct_label_dict, fw)
    return correct_label_dict

def onehot_mapping(char):
    y_onehot = [0] * len(var_list)
    char_idx = var_list.index(char)
    y_onehot[char_idx] = 1
    return y_onehot

def gen_training_data(correct_label_dict, output_path):
    with open(output_path, 'w') as fw:
        for _, val in correct_label_dict.items():
            for item in val:
                img_path, char = item
                y_onehot = onehot_mapping(char)  # dim = [36]
                y_onehot = list(map(str, y_onehot))
                y_onehot = '\t'.join(y_onehot)
                im = cv2.imread(img_path, 0).reshape([-1])  # single channel, dim = [24, 16] -> [384]
                im = list(map(str, im))
                im = '\t'.join(im)
                fw.write('{}\t'.format(char))
                fw.write('{}\t'.format(y_onehot))
                fw.write('{}\n'.format(im))

def gen_X_y_pkl(data_path, out_dir='./output/'):
    char_dict = dict()
    X = []
    y = []
    with open(data_path, 'r') as fr:
        for line in fr:
            line = line.strip('\n').split('\t')
            char = line[0]
            if char in char_dict:
                char_dict[char] += 1
            else:
                char_dict[char] = 0
            tmp_y = list(map(int, line[1:37]))
            y.append(tmp_y)
            tmp_X = list(map(int, line[37:]))
            tmp_X = [1 if pixel > 200 else 0 for pixel in tmp_X]  # convert to 0/1 for each pixel
            X.append(tmp_X)
    X = np.asarray(X)
    y = np.asarray(y)
    with open(out_dir + 'char_dict.json', 'w') as fw:
        json.dump(char_dict, fw)
    with open(out_dir + 'train_X.pkl', 'wb') as fw:
        pickle.dump(X, fw)
    with open(out_dir + 'train_y.pkl', 'wb') as fw:
        pickle.dump(y, fw)


if __name__ == '__main__':

    # use 10 images as example
    
    google_jpg_dict = gen_jpg_dict('./output/google_jpg_dict.json', 10)

    baidu_confirm_dict = gen_confirm_dict('./output/baidu_confirm_dict.json', 10)

    correct_label_dict = gen_correct_label_dict(google_jpg_dict, baidu_confirm_dict, './output/correct_label_dict.json', 10)

    gen_training_data(correct_label_dict, './output/train_data_all.txt')

    gen_X_y_pkl('./output/train_data_all.txt')