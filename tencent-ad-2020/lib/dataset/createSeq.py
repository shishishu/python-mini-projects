#!/usr/bin/env python
#coding=utf-8
# @file  : creative_sequence
# @time  : 5/17/2020 7:40 PM
# @author: shishishu

import os
from conf import config
from collections import defaultdict
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


class CreateSeq:
    def __init__(self, input_dir, ad_domains, task_type='train'):
        self.input_dir = input_dir
        self.ad_domains = ad_domains
        self.task_type = task_type
        self.output_dir = os.path.join(input_dir, self.task_type)
        self.corpus_dir = os.path.join(input_dir, 'corpus')
        FileOperation.safe_mkdir(self.output_dir)
        FileOperation.safe_mkdir(self.corpus_dir)
        # self.click_log_dict = self.parse_click_log()

    def parse_click_log(self, ad_domain):
        click_log_dict = defaultdict(lambda: list())
        click_log_path = os.path.join(self.input_dir, self.task_type, 'click_log_clean.csv')
        # time,user_id,creative_id,click_times,ad_id,product_id,product_category,advertiser_id,industry
        with open(click_log_path, 'r', encoding='utf-8') as fr:
            # next(fr)  # skip header
            header = fr.readline().strip('\n').split(',')
            logging.info('head line is: {}'.format(header))
            time_index = header.index('time')  # 0
            user_index = header.index('user_id')  # 1
            click_index = header.index('click_times')  # 3
            domain_index = header.index(ad_domain)
            for line_idx, line in enumerate(fr):
                if (line_idx + 1) % 10 ** 6 == 0:
                    logging.info("current line idx in reading is: {}".format(line_idx + 1))
                line = line.strip('\n').split(',')
                op_time = int(line[time_index])
                user_str = 'user_' + line[user_index]
                domain_str = ad_domain + '_' + line[domain_index]
                click_times = int(line[click_index])
                click_log_dict[user_str].append([domain_str, op_time, click_times])
        sorted_dict = dict()
        for user, item in click_log_dict.items():
            sorted_item = sorted(item, key=lambda x: x[1], reverse=False)
            sorted_dict[user] = sorted_item
        del click_log_dict
        return sorted_dict

    def gene_sequence(self):
        for ad_domain in self.ad_domains:
            click_log_dict = self.parse_click_log(ad_domain)
            users = list(click_log_dict.keys())
            logging.info('current ad domain is: {}'.format(ad_domain))
            total_cnt = 0
            test_path = os.path.join(self.output_dir, 'click_log_seq_' + ad_domain + '.txt')
            corpus_dir = os.path.join(self.corpus_dir, ad_domain)
            FileOperation.safe_mkdir(corpus_dir)
            corpus_path = os.path.join(corpus_dir, 'click_log_seq_' + self.task_type + '.corpus')
            with open(test_path, 'w') as fw_test, open(corpus_path, 'w') as fw_corpus:
                for user in users:
                    total_cnt += 1
                    if total_cnt % 10 ** 5 == 0:
                        logging.info("current progress is: {}".format(total_cnt))
                    item = click_log_dict[user]
                    data = CreateSeq.dict2line(user, item)
                    fw_test.write(' '.join(data) + '\n')
                    domain_list = list(zip(*item))[0]
                    fw_corpus.write(' '.join(domain_list) + '\n')
                logging.info('total user cnt is: {}'.format(total_cnt))
        return

    @staticmethod
    def dict2line(key, val):
        data = [str(key)]
        val_str = list(map(lambda x: ':'.join(list(map(str, x))), val))  # ['1:2:3', '4:5:6']
        data.extend(val_str)
        return data

    @staticmethod
    def line2dict(line):
        line = line.strip('\n').split(' ')
        user_str = line[0]
        click_item_list = list(map(lambda x: x.split(':'), line[1:]))  # list in list [creative_str, op_time, click_times]
        return user_str, click_item_list


if __name__ == '__main__':

    createSeq = CreateSeq(
        input_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        # ad_domains=['creative_id', 'ad_id', 'advertiser_id'],
        ad_domains=['product_id', 'product_category'],
        task_type='test'
    )
    createSeq.gene_sequence()

