#!/usr/bin/env python
#coding=utf-8
# @file  : dataMap
# @time  : 6/2/2020 7:51 PM
# @author: shishishu


import os
import pickle
import numpy as np
import pandas as pd
import random
from conf import config
from lib.utils.fileOperation import FileOperation
from lib.dataset.createSeq import CreateSeq
from lib.dataset.dataUtils import DataUtils

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'dataset.log'),
                    filemode='a')


class DataMap:

    def __init__(self, embedding_size, max_sen_len, input_dir, pred_domain, ad_domains, task_type, model_type='rnn', data_version='001'):
        self.embedding_size = embedding_size
        self.max_sen_len = max_sen_len
        self.pred_domain = pred_domain
        self.ad_domains = ad_domains
        self.task_type = task_type
        self.resample_thresh = config.EDA_CLIP_DICT['user_record_10th']
        self.input_dir = os.path.join(input_dir, self.task_type)
        # self.seq_data_path = os.path.join(input_dir, self.task_type, 'click_log_seq.txt')
        self.onehot_len = len(config.LABEL_TYPE_DICT[self.pred_domain])
        self.onehot_cols = ['onehot_' + str(idx) for idx in range(self.onehot_len)]
        self.model_type = model_type
        self.output_dir = os.path.join(config.DATA_DIR, self.model_type, data_version)
        FileOperation.safe_mkdir(self.output_dir)

    def gene_data_map(self):
        df_data = pd.DataFrame()
        for idx, ad_domain in enumerate(self.ad_domains):
            if idx == 0:
                df_data = self.parse_feas(ad_domain)
            else:
                tmp_df = self.parse_feas(ad_domain)
                df_data = pd.merge(df_data, tmp_df, on='user_id')
        logging.info('shape of df_data after ad domains is: {}'.format(df_data.shape))

        if self.task_type == 'train':
            df_user = self.parse_labels()
            df_user = df_user.sample(frac=1.0, random_state=42)  # shuffle
            df_user_tr = df_user.sample(frac=0.9, random_state=7)
            df_user_va = df_user[~df_user.index.isin(df_user_tr.index)]
            df_user_va_1 = df_user_va.sample(frac=0.5, random_state=7)
            df_user_va_2 = df_user_va[~df_user_va.index.isin(df_user_va_1.index)]
            df_tr = pd.merge(df_data, df_user_tr, on='user_id')
            df_tr = df_tr.sample(frac=1.0, random_state=42)
            # df_tr.drop(columns=['is_resample'], inplace=True)
            logging.info('shape of df_tr is: {}'.format(df_tr.shape))
            FileOperation.save_csv(df_tr, os.path.join(self.output_dir, 'tr_' + self.pred_domain + '.txt'), ' ', False)
            del df_tr
            df_va_1 = pd.merge(df_data, df_user_va_1, on='user_id')
            df_va_1 = df_va_1.sample(frac=1.0, random_state=42)
            df_va_2 = pd.merge(df_data, df_user_va_2, on='user_id')
            df_va_2 = df_va_2.sample(frac=1.0, random_state=42)
            # df_va_1.drop(columns=['is_resample'], inplace=True)
            # df_va_2 = pd.merge(df_data[df_data['is_resample'] == 0], df_user_va_2, on='user_id')
            # df_va_2 = df_va_2.sample(frac=1.0, random_state=42)
            # df_va_2.drop(columns=['is_resample'], inplace=True)
            logging.info('shape of df_va_1 is: {}'.format(df_va_1.shape))
            logging.info('shape of df_va_2 is: {}'.format(df_va_2.shape))
            FileOperation.save_csv(df_va_1, os.path.join(self.output_dir, 'va_1_' + self.pred_domain + '.txt'), ' ', False)
            FileOperation.save_csv(df_va_2, os.path.join(self.output_dir, 'va_2_' + self.pred_domain + '.txt'), ' ', False)
            del df_va_1, df_va_2
        else:
            cols = list(df_data.columns)
            cols.extend(self.onehot_cols)
            df_data = df_data.reindex(columns=cols)  # add cols
            df_data.loc[:, self.onehot_cols] = [0] * self.onehot_len  # fill dummy in test
            # df_data.drop(columns=['is_resample'], inplace=True)
            logging.info('shape of df_te is: {}'.format(df_data.shape))
            FileOperation.save_csv(df_data, os.path.join(self.output_dir, 'te_' + self.pred_domain + '.txt'), ' ', False)
        return

    def parse_feas(self, ad_domain='creative_id'):
        logging.info('current ad domain is: {}'.format(ad_domain))
        seq_data_path = os.path.join(self.input_dir, 'click_log_seq_' + ad_domain + '.txt')
        data_list = []
        creative_dict = DataMap.load_map_file(self.embedding_size, ad_domain)
        with open(seq_data_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                data = []
                user_str, click_item_list = CreateSeq.line2dict(line)
                user_id = int(user_str.split('_')[1])
                creative_strs = list(zip(*click_item_list))[0]
                # CLS:1, UNK:2
                creative_idx_ori = [creative_dict.get(creative_str, str(2)) for creative_str in creative_strs]  # UNK
                if self.model_type in ['txs', 'txs2']:
                    creative_idx_ori.insert(0, str(1))  # CLS
                creative_len_ori = len(creative_idx_ori)
                creative_len, creative_idx = DataMap.padding_seq(creative_idx_ori, self.max_sen_len)
                data.append(user_id)
                # data.append(0)  # origin
                data.extend(creative_idx)
                data.append(creative_len)
                data_list.append(data)
                # if self.task_type == 'train' and self.resample_time > 0 and creative_len_ori > self.resample_thresh:
                #     start_idx = 0 if self.model_type != 'txs' else 1
                #     for _ in range(self.resample_time):
                #         tmp_data = []
                #         random_index = random.randint(start_idx, creative_len_ori - 1)
                #         creative_idx_resample = creative_idx_ori.copy()
                #         creative_idx_resample.pop(random_index)
                #         creative_len, creative_idx = DataMap.padding_seq(creative_idx_resample, self.max_sen_len)
                #         tmp_data.append(user_id)
                #         tmp_data.append(1)  # resample
                #         tmp_data.extend(creative_idx)
                #         tmp_data.append(creative_len)
                #         data_list.append(tmp_data)
        cols = ['user_id']
        idx_cols = [ad_domain + '_idx_' + str(idx) for idx in range(self.max_sen_len)]
        cols.extend(idx_cols)
        cols.append(ad_domain + '_len')
        df_data = pd.DataFrame(data=data_list, columns=cols)
        logging.info('shape of df_data is: {}'.format(df_data.shape))
        return df_data

    def parse_labels(self):
        user_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'user.csv')
        df_user = FileOperation.load_csv(user_path)
        df_user = df_user[['user_id', self.pred_domain]]  # user_id, age/gender
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        df_user[self.onehot_cols] = df_user.apply(
            lambda row: DataUtils.gene_onehot(self.onehot_len, row[self.pred_domain]),
            axis=1,
            result_type='expand'
        )
        df_user.drop(columns=self.pred_domain, inplace=True)
        return df_user

    @staticmethod
    def padding_seq(word_idx_ori, max_sen_len):
        word_idx = word_idx_ori.copy()  # list
        if len(word_idx) <= max_sen_len:
            sen_len = len(word_idx)
            word_idx = word_idx + [str(0)] * (max_sen_len - sen_len)  # padding with 0
        else:
            sen_len = max_sen_len
            word_idx = word_idx[:max_sen_len]
        return str(sen_len), word_idx

    @staticmethod
    def load_map_file(embedding_size, ad_domain='creative_id'):
        model_dir = os.path.join(config.MODEL_DIR, 'w2v_' + str(embedding_size)) + '_' + ad_domain
        # load word-index map file
        with open(os.path.join(model_dir, 'word.map'), 'r', encoding='utf-8') as fr:
            lines = [line.strip('\n').split('\t') for line in fr]  # both word and index are loaded as str
        word_dict = dict(lines)  # dict: word-index
        return word_dict


if __name__ == '__main__':

    # DataMap.w2v_map_setup(128)
    dataMap = DataMap(
        embedding_size=64,
        max_sen_len=config.MAX_SEN_LEN,
        input_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        pred_domain='age',
        ad_domains=['creative_id', 'advertiser_id', 'product_id'],
        task_type='test',
        model_type='rnn3',
        data_version='002'
    )
    dataMap.gene_data_map()