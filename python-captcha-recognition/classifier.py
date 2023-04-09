#!/usr/bin/env python
#coding=utf-8
"""
Created on Sat, 4 May 2019
@author: Nano Zhou
"""

import numpy as np 
import pickle
import cv2
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from img_google_ocr import *


var_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',\
    'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def mlp_algo(X_train, y_train, X_test, y_test, hidden_layers, num_iter, model_dir='./output/'):
    print('active mlp algo...')
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='relu', max_iter=num_iter)
    clf.fit(X_train, y_train)
    joblib.dump(clf, open(model_dir + 'mlp.model', 'wb'))
    clf_accuracy(clf, X_train, y_train, X_test, y_test)
    return clf

def clf_accuracy(clf, X_train, y_train, X_test, y_test):
    pred_train = np.argmax(clf.predict_proba(X_train), axis=1)
    true_train = np.argmax(y_train, axis=1)
    pred_test = np.argmax(clf.predict_proba(X_test), axis=1)
    true_test = np.argmax(y_test, axis=1)
    print('training accuracy on single char is: ', accuracy_score(true_train, pred_train))
    print('test accuracy on single char is: ', accuracy_score(true_test, pred_test))

def clf_predict(clf, test_data):
    pred_result = ''
    num_char = len(test_data)
    for i in range(num_char):
        prob = clf.predict_proba(test_data[i])
        pred_idx = np.argmax(prob)
        pred_char = var_list[pred_idx]
        pred_result += pred_char
    return pred_result

def cvt_img_data(img_path, file_type='jpg', char_number=4):
    # img_path could be jpg file path or base64 data
    if file_type == 'jpg':
        im = cv2.imread(img_path, 0)  # convert to gray image directly
    else:
        im_arr = np.fromstring(img_path, np.uint8)  # img_path = img_data here
        im = cv2.imdecode(im_arr, cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im_32 = np.array(im, dtype=np.int32)
    im_32 = remove_line(im_32, 20)
    im = np.array(im_32, dtype=np.uint8)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 1)
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im[im <= 200] = 0
    im[im > 200] = 1
    split_length = int(im.shape[1]/char_number)
    assert split_length == 16, 'wrong split'
    test_data = []
    for i in range(char_number):
        tmp_im = im[:, split_length*i:split_length*(i+1)]
        tmp_im = tmp_im.reshape(1, -1)
        test_data.append(tmp_im)
    return test_data

def cal_pred_accuracy(clf, test_dir='./golden_set/'):
    total_num = 0
    correct_num = 0
    img_names = glob.glob(test_dir + '/*.jpg')
    for img_path in img_names:
        total_num += 1
        _, img_name_stem, _ = get_dir_file_name(img_path)
        real_tags = img_name_stem.lower()
        test_data = cvt_img_data(img_path)
        pred_result = clf_predict(clf, test_data)
        # print('real is {} -> pred is {}, flag is {}'.format(real_tags, pred_result, real_tags==pred_result))
        if real_tags == pred_result:
            correct_num += 1
    print('total accuracy on golden set is: {}'.format(correct_num / total_num))


if __name__ == '__main__':
    
    X = pickle.load(open('./output/train_X.pkl', 'rb'))
    y = pickle.load(open('./output/train_y.pkl', 'rb'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    clf = mlp_algo(X_train, y_train, X_test, y_test, (32, 32), 10)

    cal_pred_accuracy(clf, test_dir='./golden_set/')
    
    
    start = time.time()
    clf = joblib.load('./output/mlp.model')
    print('time cost in loading model is: ', time.time()-start)

    start = time.time()
    img_path = './golden_set/C4j2.jpg'
    test_data = cvt_img_data(img_path)
    print('input is: ', get_dir_file_name(img_path)[1])
    pred_result = clf_predict(clf, test_data)
    print('pred is: ', pred_result)
    print('time cost in single prediction is: ', time.time()-start)