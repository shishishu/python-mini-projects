#!/usr/bin/env python
#coding=utf-8
# @file  : config
# @time  : 5/17/2020 7:40 PM
# @author: shishishu

import os

master_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATA_DIR = os.path.join(master_path, 'data')
MODEL_DIR = os.path.join(master_path, 'model')
LOG_DIR = os.path.join(master_path, 'log')


AGE_DIST_DICT = {1.0: 0.0391, 2.0: 0.1659, 3.0: 0.2255, 4.0: 0.1673, 5.0: 0.1452, 6.0: 0.113, 7.0: 0.0741, 8.0: 0.0355, 9.0: 0.0216, 10.0: 0.0128}
AGE_BIG_GROUP = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5, 6],
    4: [7, 8, 9, 10]
}
AGE_BIG_GROUP_INV = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4, 10: 4}

GENDER_DIST_DICT = {1.0: 0.6696, 2.0: 0.3304}

LABEL_TYPE_DICT = {
    'age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gender': [1, 2]
}

EDA_CLIP_DICT = {
    'click_times_99th': 2,
    'user_record_1th': 9,
    'user_record_10th': 11,
    'user_record_30th': 16,
    'user_record_50th': 24,
    'user_record_80th': 46,
    'user_record_90th': 66,
    'user_record_95th': 89,
    'user_record_99th': 156,
    'creative_record_1th': 1,
    'creative_record_99th': 140
}

# MAX_SEN_LEN = 90  # v1
MAX_SEN_LEN = 46  # v2

# common / self-total
CATEGORY_DIFF_DICT = {
    'creative_id': {'ad': [0.68, 0.64], 'click': [0.97, 0.96]},  # 2.5M
    'ad_id': {'ad': [0.71, 0.68], 'click': [0.97, 0.97]},  # 2.3M
    'advertiser_id': {'ad': [0.90, 0.89], 'click': [1.0, 1.0]},  # 53K
    'product_id': {'ad': [0.85, 0.83], 'click': [1.0, 1.0]},  # 33K, na rate = 40%
    'industry': {'ad': [0.98, 0.98], 'click': [1.0, 1.0]},  # 326, na rate = 4%
    'product_category': {'ad': [1.0, 1.0], 'click': [1.0, 1.0]}  # 18, very important
}

ONEHOT_DIM_DICT = {
    'rnn2cate': 15,
    'rnn2gender': 1,
    'rnn2cross': 30,
    'rnn2concat': 16
}