#!/usr/bin/env python
#coding=utf-8
# @file  : clean_data
# @time  : 5/22/2020 11:27 PM
# @author: shishishu

import os
import pandas as pd
from conf import config
from lib.utils.fileOperation import FileOperation

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


class DataClearner:

    def __init__(self, input_dir, output_dir, task_type='train'):
        self.task_type = task_type
        self.input_dir = os.path.join(input_dir, self.task_type)
        self.output_dir = os.path.join(output_dir, self.task_type)
        FileOperation.safe_mkdir(self.output_dir)

    def clean_data(self):
        click_path = os.path.join(self.input_dir, 'click_log.csv')
        ad_path = os.path.join(self.input_dir, 'ad.csv')
        df_click = FileOperation.load_csv(click_path)
        logging.info('shape of df_click is: {}'.format(df_click.shape))
        df_ad = FileOperation.load_csv(ad_path)
        df_ad = df_ad.applymap(lambda x: int(x) if x != '\\N' else 0)  # convert to int
        logging.info('shape of df_ad is: {}'.format(df_ad.shape))
        df_click = pd.merge(df_click, df_ad, on='creative_id')
        click_cols = list(df_click.columns)
        logging.info('shape of post df_click is: {}'.format(df_click.shape))
        del df_ad

        click_times_99th = config.EDA_CLIP_DICT['click_times_99th']
        df_click['click_times'] = df_click['click_times'].apply(lambda x: min(x, click_times_99th))

        df_click_user = df_click['user_id'].value_counts().to_frame().reset_index()
        df_click_user.columns = ['user_id', 'user_record']
        df_click_expand = pd.merge(df_click, df_click_user, on='user_id', how='left')
        logging.info('shape of df_click_expand is: {}'.format(df_click_expand.shape))
        del df_click

        user_record_99th = config.EDA_CLIP_DICT['user_record_99th']
        df_part1 = df_click_expand[df_click_expand['user_record'] <= user_record_99th]
        logging.info('shape of df_part1 is: {}'.format(df_part1.shape))
        df_part2 = pd.DataFrame()
        for idx, row in df_click_user[df_click_user['user_record'] > user_record_99th].iterrows():
            tmp_df = df_click_expand[df_click_expand['user_id'] == row['user_id']]
            tmp_df = tmp_df.sample(n=user_record_99th, random_state=7)
            df_part2 = pd.concat([df_part2, tmp_df], axis=0)
        logging.info('shape of df_part2 is: {}'.format(df_part2.shape))
        del df_click_user

        df = pd.concat([df_part1, df_part2], axis=0)
        logging.info('shape of filter df is: {}'.format(df.shape))
        del df_part1, df_part2
        logging.info('num of unique user is: {}'.format(len(set(df['user_id']))))

        clean_path = os.path.join(self.output_dir, 'click_log_clean.csv')
        FileOperation.save_csv(df[click_cols], clean_path)  # keep click cols only
        return


if __name__ == '__main__':

    dataCleaner = DataClearner(
        input_dir=os.path.join(config.DATA_DIR, 'raw'),
        output_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        task_type='test'
    )
    dataCleaner.clean_data()
