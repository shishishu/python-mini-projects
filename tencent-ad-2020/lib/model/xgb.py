#!/usr/bin/env python
#coding=utf-8
# @file  : xgb
# @time  : 5/19/2020 11:07 PM
# @author: shishishu

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lib.utils.fileOperation import FileOperation
from lib.utils.utils import collect_log_content, collect_log_key_content, sample_df_pipeline, calculate_dist
from conf import config

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'xgb.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class XGB:

    def __init__(self, input_dir, xgb_params, sample_strategy, pred_domain='age', task_type='train', data_version='001', model_version='0010'):
        logging.info('\n')
        logging.info('data_version is: {}, pred_domain is: {}, model_version is: {}'.format(data_version, pred_domain, model_version))

        self.input_dir = input_dir
        self.pred_domain = pred_domain
        model_name = 'xgb_pred_' + self.pred_domain + '_' + model_version
        self.model_dir = os.path.join(config.MODEL_DIR, model_name)
        FileOperation.safe_mkdir(self.model_dir)
        self.xgb_params = xgb_params
        self.xgb_params['num_class'] = 10 if self.pred_domain == 'age' else 2
        FileOperation.save_json(self.xgb_params, os.path.join(self.model_dir, 'xgb_params.json'))
        self.sample_strategy = sample_strategy[self.pred_domain]
        FileOperation.save_json(self.sample_strategy, os.path.join(self.model_dir, 'sample_strategy.json'))
        self.model_path = os.path.join(self.model_dir, 'xgb.pickle.dat')
        self.task_type = task_type
        self.data_version = data_version
        self.log_dict = dict()
        self.log_path = os.path.join(self.model_dir, 'log_dict.json')

    def parse_input_train(self, data_path):
        df_data = FileOperation.load_csv(data_path)  # user_id, fea_1, fea_2...
        logging.info('shape of df_data is: {}'.format(df_data.shape))
        user_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'user.csv')
        df_user = FileOperation.load_csv(user_path)
        df_user = df_user[['user_id', self.pred_domain]]  # user_id, age/gender
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        df_data = pd.merge(df_data, df_user, on='user_id', how='left')
        logging.info('shape of df_data after merge df_user is: {}'.format(df_data.shape))
        df_data = df_data.sample(frac=1.0, random_state=7)  # shuffle
        dist_dict = calculate_dist(df_data, self.pred_domain)
        logging.info('original dist of {} is: {}'.format(self.pred_domain, dist_dict))
        collect_log_content(self.log_dict, 'original dist of {} is: {}'.format(self.pred_domain, dist_dict), self.log_path)
        del df_user

        df_tr = df_data.sample(frac=0.9, random_state=7)
        df_va = df_data[~df_data.index.isin(df_tr.index)]
        logging.info('original shape of df_tr is: {}'.format(df_tr.shape))
        logging.info('original shape of df_va is: {}'.format(df_va.shape))

        df_tr_sample = sample_df_pipeline(df_tr, self.pred_domain, self.sample_strategy)
        logging.info('shape of df_tr_sample is: {}'.format(df_tr_sample.shape))
        tr_dist_dict = calculate_dist(df_tr_sample, self.pred_domain)
        logging.info('sampled tr dist of {} is: {}'.format(self.pred_domain, tr_dist_dict))
        collect_log_content(self.log_dict, 'sampled tr dist of {} is: {}'.format(self.pred_domain, tr_dist_dict), self.log_path)
        del df_tr

        df_va_sample = sample_df_pipeline(df_va, self.pred_domain, self.sample_strategy)
        logging.info('shape of df_va_sample is: {}'.format(df_va_sample.shape))
        va_dist_dict = calculate_dist(df_va_sample, self.pred_domain)
        logging.info('sampled va dist of {} is: {}'.format(self.pred_domain, va_dist_dict))
        collect_log_content(self.log_dict, 'sampled va dist of {} is: {}'.format(self.pred_domain, va_dist_dict), self.log_path)

        data_dict = dict()
        data_dict['tr_sample_data'] = self.split_data(df_tr_sample, 'train')
        data_dict['va_sample_data'] = self.split_data(df_va_sample, 'train')
        data_dict['va_data'] = self.split_data(df_va, 'train')
        del df_tr_sample, df_va_sample, df_va
        return data_dict

    def parse_input_test(self, data_path):
        df_data = FileOperation.load_csv(data_path)  # user_id, fea_1, fea_2...
        logging.info('shape of df_data is: {}'.format(df_data.shape))
        user_data, X, y = self.split_data(df_data, 'test')
        del df_data
        return user_data, X, y

    def split_data(self, df_data, task_type):
        user_data = df_data.values[:, 0]
        logging.info('shape of user_data is: {}'.format(user_data.shape))
        if task_type == 'train':
            X = df_data.values[:, 1:-1]
            Y = df_data.values[:, -1]
            label_encoder = LabelEncoder().fit(Y)
            y = label_encoder.transform(Y)
        else:
            X = df_data.values[:, 1:]
            y = None
        return user_data, X, y

    def xgb_train_eval(self):
        data_path = os.path.join(self.input_dir, self.task_type, self.data_version, 'dataset.csv')
        data_dict = self.parse_input_train(data_path)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
        _, X_train, y_train = data_dict['tr_sample_data']
        logging.info('shape of training input is: {}'.format(X_train.shape))
        collect_log_content(self.log_dict, 'shape of training input is: {}'.format(X_train.shape), self.log_path)
        clf = xgb.XGBClassifier(**self.xgb_params)
        logging.info('Start training xgb model...')
        clf.fit(X_train, y_train)
        logging.info('xgb model training is done...')
        pickle.dump(clf, open(self.model_path, 'wb'))
        logging.info('xgb model is saved...')

        # hidden_layer_sizes = [128, 128, 128, 32]
        # hidden_layer_sizes = [256, 512, 512, 512, 256, 64]
        # clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam')
        # collect_log_key_content(self.log_dict, 'hidden_layer_sizes', hidden_layer_sizes, self.log_path)
        #
        # logging.info('Start training mlp model...')
        # clf.fit(X_train, y_train)
        # logging.info('mlp model training is done...')
        # pickle.dump(clf, open(self.model_path, 'wb'))
        # logging.info('mlp model is saved...')

        _, X_valid_sample, y_valid_sample = data_dict['va_sample_data']
        df_res = self.pred_module(X_valid_sample, y_valid_sample)
        eval_name = self.pred_domain + '_eval_detail_sample.csv'
        FileOperation.save_csv(df_res, os.path.join(self.model_dir, eval_name))
        acc = round(accuracy_score(df_res['y_true'], df_res['y_pred']), 4)
        logging.info('acc eval with sampling in domain : {} is: {}'.format(self.pred_domain, acc))
        collect_log_content(self.log_dict, 'acc eval with sampling in domain: {} is: {}'.format(self.pred_domain, acc), self.log_path)

        _, X_valid, y_valid = data_dict['va_data']
        df_res = self.pred_module(X_valid, y_valid)
        eval_name = self.pred_domain + '_eval_detail.csv'
        FileOperation.save_csv(df_res, os.path.join(self.model_dir, eval_name))
        acc = round(accuracy_score(df_res['y_true'], df_res['y_pred']), 4)
        logging.info('acc eval in domain : {} is: {}'.format(self.pred_domain, acc))
        collect_log_content(self.log_dict, 'acc eval in domain: {} is: {}'.format(self.pred_domain, acc), self.log_path)
        return

    def xgb_infer(self):
        data_path = os.path.join(self.input_dir, 'test', self.data_version, 'dataset.csv')
        user_data, X, y = self.parse_input_test(data_path)
        df_res = self.pred_module(X, y)
        pred_name = self.pred_domain + '_pred_detail.csv'
        FileOperation.save_csv(df_res, os.path.join(self.model_dir, pred_name))
        export_name = self.pred_domain + '_export.csv'
        df_exp = self.export_prediction(user_data, df_res)
        FileOperation.save_csv(df_exp, os.path.join(self.model_dir, export_name))
        return

    def pred_module(self, X, y):
        y_true = y if y is not None else np.asarray([-1] * X.shape[0])  # fill dummy val
        logging.info('shape of input is: {}'.format(X.shape))
        clf = None
        if os.path.exists(self.model_path):
            clf = pickle.load(open(self.model_path, 'rb'))
            logging.info('xgb model is loaded...')
        else:
            logging.error('no xgb model found, exit...')
            sys.exit(1)
        y_pred = clf.predict(X)
        y_pred_prob = clf.predict_proba(X)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        dataset = np.concatenate([y_true, y_pred, y_pred_prob], axis=1)
        cols = ['y_true', 'y_pred']
        pred_type_cols = ['pred_prob_' + str(idx) for idx in range(y_pred_prob.shape[1])]
        cols.extend(pred_type_cols)
        df_res = pd.DataFrame(data=dataset, columns=cols)
        df_res[['y_true', 'y_pred']] = df_res[['y_true', 'y_pred']].applymap(int)
        df_res[pred_type_cols] = df_res[pred_type_cols].applymap(float)
        logging.info('shape of df result is: {}'.format(df_res.shape))
        del dataset
        return df_res

    def export_prediction(self, user_data, df_res):
        pred_label = 'predicted_' + self.pred_domain
        df_res[pred_label] = df_res['y_pred'].apply(lambda x: int(x+1))  # origin starts from 1
        df_user = pd.DataFrame(data={'user_id': user_data, pred_label: list(df_res[pred_label].values)}, columns=['user_id', pred_label], dtype='int')
        logging.info('shape of df_user after merge with predictions isL {}'.format(df_user.shape))
        pred_dist_dict = calculate_dist(df_user, pred_label)
        logging.info('pred dist of {} is: {}'.format(self.pred_domain, pred_dist_dict))
        collect_log_content(self.log_dict, 'pred dist of {} is: {}'.format(self.pred_domain, pred_dist_dict), self.log_path)
        return df_user

    @staticmethod
    def find_hf_item(item_list):
        return max(set(item_list), key=item_list.count)

    @staticmethod
    def find_max_prob_after_avg(lists):
        arr = np.asarray(lists)
        arr_avg = np.mean(arr, axis=0)
        return np.argmax(arr_avg)


if __name__ == '__main__':

    xgb_params = {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_delta_step': 1,
        'scale_pos_weight': 1,
        'objective': 'multi:softmax',
        'n_jobs': 12,
        'random_state': 7,
        'silent': False
    }
    sample_strategy = {
        'age': {
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 1.0,
            8: 1.0,
            9: 1.0,
            10: 1.0
        },
        'gender': {
            1: 1.0,
            2: 1.2
        }
    }

    xgber = XGB(
        input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
        xgb_params=xgb_params,
        sample_strategy=sample_strategy,
        pred_domain='gender',
        task_type='train',
        data_version='003',
        model_version='0033'
    )

    xgber.xgb_train_eval()
    xgber.xgb_infer()