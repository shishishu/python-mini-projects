#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 19 May 2019
@author: Nano Zhou
"""

import pytesseract
import os
import multiprocessing
import cv2
import numpy as np
import argparse
import ast
import re
import functools
import operator
from lxml import etree
from fontTools.ttLib import TTFont
from Woff2Text import Woff2Text, WoffUrl2Text
import time


class ParseText:

    def __init__(self, w2t, num_process, chunk_size, resize_size, ocr_config, fill_color, line_ratio):
        self.w2t = w2t  # pass Woff2Text class as one attribute
        self.num_process = num_process
        self.chunk_size = chunk_size
        self.resize_size = resize_size
        self.ocr_config = ocr_config
        self.fill_color = fill_color
        self.line_ratio = line_ratio
        self.img_method = 'm1'  # use method 1 as default

    def create_img(self, item):
        key = item[0]
        content = item[1]
        im = np.zeros([content['height'], content['width']], dtype=np.uint8)
        if self.img_method == 'm1':
            for sub_data in content['data']:
                ParseText.fill_contour(im, sub_data, self.fill_color)
        else:
            line_width = max(int(content['width'] * self.line_ratio), 1)  # ensure line_width >= 1
            for sub_data in content['data']:
                ParseText.line_contour(im, sub_data, self.fill_color, line_width)
            if self.img_method == 'm2':
                ParseText.reconstruct_contour(im, self.fill_color)
        im = cv2.flip(im, 0)  # flip in x-axis
        im = cv2.resize(im, (self.resize_size, self.resize_size))
        return key, im
    
    def char_pair(self, img_pair):
        key = img_pair[0]
        im = img_pair[1]
        char = pytesseract.image_to_string(im, config=self.ocr_config)
        return key, char
    
    def char_pair_chunk(self, img_pair_chunk):
        keys, ims = list(zip(*img_pair_chunk))
        im_hstack = np.hstack(ims)
        chars = pytesseract.image_to_string(im_hstack, config=self.ocr_config)  # recognize several chars per pass
        if len(chars) == len(keys):
            return list(zip(keys, chars))  # [(key, char), (key, char)...]
        else:
            return [self.char_pair(img_pair) for img_pair in img_pair_chunk]  # recognizre one char per pass

    def ocr_pipeline(self, source):
        pool = multiprocessing.Pool(processes=self.num_process)
        img_pairs = pool.map(self.create_img, source)  # change based on m1/m2/m3, return list of (key, im) pairs
        # [[(key, im), (key, im)...], [(key, im), (key, im)...]...]
        img_pair_chunks = [img_pairs[i: i + self.chunk_size] for i in range(0, len(img_pairs), self.chunk_size)]
        char_pair_chunks = pool.map(self.char_pair_chunk, img_pair_chunks)
        char_pairs = functools.reduce(operator.iconcat, char_pair_chunks, [])  # flatten chunks
        if self.img_method != 'm3':
            re_char_pairs = pool.map(ParseText.re_char_pair, char_pairs)  # do not regularize at m3
            return dict(re_char_pairs)
        return dict(char_pairs)

    def recognize_text(self):
        results = dict()
        unparse_items_1 = dict()
        re_pairs_1 = self.ocr_pipeline(self.w2t.text.items())
        for key, val in re_pairs_1.items():
            if val != '':
                results[key] = val
            else:
                unparse_items_1[key] = self.w2t.text[key]  # (key, item)

        if len(results) != len(self.w2t.text):
            self.img_method = 'm2'
            unparse_items_2 = dict()
            re_pairs_2 = self.ocr_pipeline(unparse_items_1.items())
            for key, val in re_pairs_2.items():
                if val != '':
                    results[key] = val
                else:
                    unparse_items_2[key] = self.w2t.text[key]
        
        if len(results) != len(self.w2t.text):
            self.img_method = 'm3'
            char_pairs_3 = self.ocr_pipeline(unparse_items_2.items())
            for key, val in char_pairs_3.items():
                results[key] = val
        
        assert len(results) == len(self.w2t.text), 'some chars are missed in parsing...'

        # print('parsed results is: ', results)
        return results

    @staticmethod
    def fill_contour(im, sub_data, fill_color):
        new_data = np.array(sub_data, dtype=np.int32).reshape(1, -1, 2)  # reshape is important
        cv2.fillPoly(im, new_data, fill_color)
    
    @staticmethod
    def line_contour(im, sub_data, fill_color, line_width):
        new_data = np.array(sub_data, dtype=np.int32).reshape(1, -1, 2)
        cv2.polylines(im, new_data, isClosed=1, color=fill_color, thickness=line_width)
    
    @staticmethod
    def reconstruct_contour(im, fill_color):  # find and fill (only adapt to simple character!!!)
        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy.reshape(-1, 4)
        h_level = [[i, dict()] for i in range(len(contours))]
        h_level = dict(h_level)
        sys_level = 0  # it starts from 0 always at idx = 0
        for i in range(len(contours)):
            if not 'level' in h_level[i]:
                tmp = []
                h_level[i]['level'] = sys_level
                tmp.append(i)
                j = i
                while True:
                    j = hierarchy[j, 0]
                    if j == -1:
                        break
                    h_level[j]['level'] = sys_level
                    tmp.append(j)
                for k in tmp:
                    k = hierarchy[k, 2]
                    if k != -1:
                        h_level[k]['level'] = sys_level
                sys_level += 1
            if h_level[i]['level'] % 2 == 0:
                cv2.drawContours(im, contours, i, fill_color, cv2.FILLED)
            else:
                cv2.drawContours(im, contours, i, 0, cv2.FILLED)  # re-fill dark, very import
    
    @staticmethod
    def re_char_pair(char_pair):
        key = char_pair[0]
        char = char_pair[1]
        char = re.sub('[a-zA-Z-+=@#&_,.:;?!()""`，。：；？！（）、……“”]+', '', char)
        return key, char


def process(woff_target, **kwargs):

    woff_class = WoffUrl2Text  # use url parsing as default

    if 'is_url' in kwargs and not kwargs['is_url']:
        woff_class = Woff2Text

    if 'margin_ratio' in kwargs:
        margin_ratio = kwargs['margin_ratio']
        w2t = woff_class(woff_target, margin_ratio)
    else:
        w2t = woff_class(woff_target)

    parsing_params = {
        'num_process': 8,
        'chunk_size': 8,
        'resize_size': 256,
        'ocr_config': ('--oem 0 --psm 8 -l chi_sim'),  # psm = 8: word
        'fill_color': 255,
        'line_ratio': 0.02
    }  # default params setting
    for key, val in kwargs.items():
        if key in parsing_params:
            parsing_params[key] = val  # update params dynamicly
    
    if parsing_params['chunk_size'] == 1:
        parsing_params['ocr_config'] = ('--oem 0 --psm 10 -l chi_sim')  # psm = 10: char

    ocrParser = ParseText(w2t, **parsing_params)
    results = ocrParser.recognize_text()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--woff_target',
        type=str,
        required=True,
        help='file path or url to woff target'
    )
    parser.add_argument(
        '--is_url',
        type=ast.literal_eval,
        default=True,
        help='use url or not'
    )
    parser.add_argument(
        '--margin_ratio',
        type=float,
        default=0.15,
        help='margin ratio between character and image edge'
    )
    parser.add_argument(
        '--num_process',
        type=int,
        default=8,
        help='num of processes in multiprocessing'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=8,
        help='num of images concatenated before ocr'
    )
    parser.add_argument(
        '--resize_size',
        type=int,
        default=256,
        help='size used in final image'
    )
    parser.add_argument(
        '--line_ratio',
        type=float,
        default=0.02,
        help='line ratio compared to image width'
    )

    FLAGS, unparsed = parser.parse_known_args()

    kwargs = {
        'is_url': FLAGS.is_url,
        'margin_ratio': FLAGS.margin_ratio,
        'num_process': FLAGS.num_process,
        'chunk_size': FLAGS.chunk_size,
        'resize_size': FLAGS.resize_size,
        'line_ratio': FLAGS.line_ratio,
    }

    start = time.time()
    results = process(FLAGS.woff_target, **kwargs)
    print('time cost in processing is: ', time.time()-start)