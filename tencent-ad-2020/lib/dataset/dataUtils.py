#!/usr/bin/env python
#coding=utf-8
# @file  : dataUtils
# @time  : 5/31/2020 4:49 PM
# @author: shishishu

import os
import pandas as pd
import numpy as np
from conf import config
from collections import Counter
from lib.utils.fileOperation import FileOperation
from joblib import Parallel, delayed

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'dataset.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DataUtils:

    @staticmethod
    def submit_result(age_model_version, gender_model_version, spec_dir):
        output_dir = os.path.join(config.DATA_DIR, 'submit', spec_dir)
        FileOperation.safe_mkdir(output_dir)
        pred_models = [age_model_version, gender_model_version]
        FileOperation.save_json(pred_models, os.path.join(output_dir, 'pred_models.json'))
        dest_path = os.path.join(output_dir, 'submission_' + spec_dir + '.csv')
        pred_age_path = os.path.join(config.MODEL_DIR, age_model_version, 'age_export.csv')
        df_age = FileOperation.load_csv(pred_age_path)
        logging.info('shape of df_age is: {}'.format(df_age.shape))
        age_max = df_age['predicted_age'].max()
        age_min = df_age['predicted_age'].min()
        assert age_min >= 1, 'wrong occurs at predicted age min...{}'.format(age_min)
        assert age_min <= 10, 'wrong occurs at predicted age max...{}'.format(age_max)

        pred_gender_path = os.path.join(config.MODEL_DIR, gender_model_version, 'gender_export.csv')
        df_gender = FileOperation.load_csv(pred_gender_path)
        logging.info('shape of df_gender is: {}'.format(df_gender.shape))
        gender_max = df_gender['predicted_gender'].max()
        gender_min = df_gender['predicted_gender'].min()
        assert gender_min >= 1, 'wrong occurs at predicted gender min...{}'.format(gender_min)
        assert gender_max <= 2, 'wrong occurs at predicted gender max...{}'.format(gender_max)

        df_submit = pd.merge(df_age, df_gender, on='user_id', how='left')
        df_submit = df_submit[['user_id', 'predicted_age', 'predicted_gender']]
        logging.info('shape of df_submit is: {}'.format(df_submit.shape))
        FileOperation.save_csv(df_submit, dest_path)
        return

    @staticmethod
    def convert_dateset(input_dir, pred_domain, task_type, data_version, output_dir):
        FileOperation.safe_mkdir(output_dir)
        onehot_len = 10 if pred_domain == 'age' else 2  # age/gender
        onehot_cols = ['onehot_' + str(idx) for idx in range(onehot_len)]
        data_path = os.path.join(input_dir, task_type, data_version, 'dataset.csv')
        df_data = FileOperation.load_csv(data_path)  # user_id, fea_0, fea_1, ...
        df_data = df_data.sample(frac=1.0, random_state=7)  # shuffle
        logging.info('shape of df_data is: {}'.format(df_data.shape))
        if task_type == 'train':
            user_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'user.csv')
            df_user = FileOperation.load_csv(user_path)
            df_user = df_user[['user_id', pred_domain]]  # user_id, age/gender
            logging.info('shape of df_user is: {}'.format(df_user.shape))
            df_user[onehot_cols] = df_user.apply(
                lambda row: DataUtils.gene_onehot(onehot_len, row[pred_domain]),
                axis=1,
                result_type='expand'
            )
            df_user.drop(columns=pred_domain, inplace=True)
            df_data = pd.merge(df_data, df_user, on='user_id', how='left')
            logging.info('shape of df_data after merge df_user is: {}'.format(df_data.shape))
            df_tr = df_data.sample(frac=0.9, random_state=7)
            logging.info('shape of df_tr is: {}'.format(df_tr.shape))
            FileOperation.save_csv(df_tr, os.path.join(output_dir, 'tr_' + pred_domain + '.txt'), ' ', False)
            df_va = df_data[~df_data.index.isin(df_tr.index)]
            logging.info('shape of df_va is: {}'.format(df_va.shape))
            FileOperation.save_csv(df_va, os.path.join(output_dir, 'va_' + pred_domain + '.txt'), ' ', False)
        else:
            cols = list(df_data.columns)
            cols.extend(onehot_cols)
            df_data = df_data.reindex(columns=cols)  # add cols
            df_data.loc[:, onehot_cols] = [0] * onehot_len  # fill dummy in test
            logging.info('shape of df_te is: {}'.format(df_data.shape))
            FileOperation.save_csv(df_data, os.path.join(output_dir, 'te_' + pred_domain + '.txt'), ' ', False)
        return

    @staticmethod
    def gene_onehot(onehot_len, target_val):
        onehot_list = [0] * onehot_len
        onehot_list[int(target_val - 1)] = 1
        return onehot_list

    @staticmethod
    def pred_result_fusion(pred_domain, model_version, pred_models, n_jobs=10):
        model_name = 'fusion_' + pred_domain + '_' + model_version
        model_dir = os.path.join(config.MODEL_DIR, model_name)
        FileOperation.safe_mkdir(model_dir)
        # pred_models: [['rnn2', '0010'], ['rnn3', '0010']]
        pred_models_fusion = list(map(lambda x: '_'.join([x[0], pred_domain, x[1]]), pred_models))
        FileOperation.save_json(pred_models_fusion, os.path.join(model_dir, 'pred_models_fusion.json'))
        dfs = []
        for idx, pred_model_name in enumerate(pred_models_fusion):
            logging.info('current model is: {}'.format(pred_model_name))
            pred_path = os.path.join(config.MODEL_DIR, pred_model_name, pred_domain + '_export.csv')
            df = FileOperation.load_csv(pred_path)
            df.columns = ['user_id', 'pred_model_' + str(idx)]
            logging.info('shape of df is: {}'.format(df.shape))
            dfs.append(df)
        df_user = dfs[0][['user_id']]
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        for pred_df in dfs:
            df_user = pd.merge(df_user, pred_df, on='user_id', how='left')
        logging.info('shape of df_user after merge is: {}'.format(df_user.shape))
        pred_cols = list(df_user.columns)[1:]
        pred_result_col = 'predicted_' + pred_domain
        def apply_counter(df1):
            df1[[pred_result_col, 'max_freq']] = df1.apply(
                lambda row: sorted(Counter(row[pred_cols]).items(), key=lambda x: x[1], reverse=True)[0], axis=1, result_type='expand'
            )
            return df1
        df_grouped = np.split(df_user, n_jobs, axis=0)
        df_user = DataUtils.apply_parallel(df_grouped, apply_counter, n_jobs)
        del df_grouped
        logging.info('shape of df_user after Counter is: {}'.format(df_user.shape))
        FileOperation.save_csv(df_user[['user_id', pred_result_col]], os.path.join(model_dir, pred_domain + '_export.csv'))
        FileOperation.save_csv(df_user, os.path.join(model_dir, pred_domain + '_export_detail.csv'))
        df_stat = df_user['max_freq'].value_counts().to_frame().reset_index()
        df_stat.columns = ['max_freq', 'cnt']
        FileOperation.save_csv(df_stat, os.path.join(model_dir, pred_domain + '_freq_stat.csv'))
        return

    @staticmethod
    def apply_parallel(df_grouped, apply_func, n_jobs):
        dfs = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(apply_func)(group) for group in df_grouped)
        return pd.concat(dfs)


if __name__ == '__main__':

    # submit result
    DataUtils.submit_result(
        age_model_version='fusion_age_0013',
        gender_model_version='rnn2_gender_0010',  # rnn_gender_0030, rnn2_gender_0010, fusion_gender_0010
        spec_dir='0622_04'
    )

    # model fusion
    # DataUtils.pred_result_fusion(
    #     pred_domain='age',
    #     model_version='0013',
    #     pred_models=[['rnn2', '0010'], ['rnn2', '0011'], ['rnn2', '0012'], ['rnn2', '0013'], ['rnn2', '0014'], ['rnn2', '0015'],
    #                  ['rnn2gender', '0010'],
    #                  ['rnn2cate', '0010'], ['rnn2cate', '0011'], ['rnn2cate', '0012'], ['rnn2cate', '0013'],
    #                  ['rnn2cross', '0010'], ['rnn2cross', '0011'],
    #                  ['rnn2concat', '0010'], ['rnn2concat', '0011'],
    #                  ['rnn3', '0010'], ['rnn3', '0020'],
    #                  ['txs2', '0010']]
    # )
    # DataUtils.pred_result_fusion(
    #     pred_domain='gender',
    #     model_version='0010',
    #     pred_models=[['rnn', '0030'], ['rnn2', '0010'], ['rnn2', '0011']]
    # )

    # convert dataset
    # DataUtils.convert_dateset(
    #     input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
    #     pred_domain='age',
    #     task_type='test',
    #     data_version='003',
    #     output_dir=os.path.join(config.DATA_DIR, 'mlp', '001')
    # )