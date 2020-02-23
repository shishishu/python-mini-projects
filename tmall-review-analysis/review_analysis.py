#!/usr/bin/env python
#coding=utf-8
# @file  : review_analysis
# @time  : 2/23/2020 4:29 PM
# @author: shishishu

import pandas as pd
import jieba.posseg as pseg
import os
import glob
import argparse
from conf import config
from collections import defaultdict, Counter, OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--shop_name', type=str, required=True, help='shop name')  # 传递参数


class ReviewAnalysis:

    def __init__(self, shop_name):
        self.data_dir = os.path.join(config.data_dir, shop_name)
        self.raw_file = os.path.join(self.data_dir, 'raw_comment_' + shop_name + '.xlsx')
        self.eda_file = os.path.join(self.data_dir, 'eda_comment_' + shop_name + '.xlsx')

    def dump_stats(self):
        df = pd.read_excel(self.raw_file)
        df['comment'].fillna("", inplace=True)
        print('raw df shape is: ', df.shape)

        # 加载停用词
        stop_words = ReviewAnalysis.load_stopwords()
        # 评论分词
        df['comment_seg'] = df['comment'].apply(lambda x: ReviewAnalysis.pseg_comment(x, stop_words))

        # 词频统计及输出
        writer = pd.ExcelWriter(self.eda_file)

        big_word_list, word_pos_dict2 = ReviewAnalysis.word_stats(df, 'all')
        df_all = pd.Series(data=big_word_list, name='top_word').to_frame()
        for key, val in word_pos_dict2.items():
            tmp = pd.Series(data=val, name=key).to_frame()
            df_all = pd.concat([df_all, tmp], axis=1)
        df_all.to_excel(writer, sheet_name='all_items', index=False)

        item_set = set(df['item_name'].values)
        df_sep = pd.DataFrame()
        for item_name in item_set:
            sub_word_list, _ = ReviewAnalysis.word_stats(df, item_name)
            tmp = pd.Series(data=sub_word_list, name=item_name).to_frame()
            df_sep = pd.concat([df_sep, tmp], axis=1)
        df_sep.to_excel(writer, sheet_name='sep_item', index=False)

        writer.save()
        return

    # 加载停用词
    @staticmethod
    def load_stopwords():
        stop_words = []
        file_paths = glob.glob(os.path.join(config.dict_dir, '*.txt'))
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    stop_words.append(line.strip('\n'))
        stop_words = set(stop_words)
        print('number of stop words is: ', len(stop_words))
        return stop_words

    # 分词（词，词性）
    @staticmethod
    def pseg_comment(content, stop_words):
        word_pairs = []
        words = pseg.cut(content)
        for w in words:
            if w.word not in stop_words:  # 去除停用词
                word_pairs.append([w.word, w.flag])
        return word_pairs

    # 词频统计
    @staticmethod
    def word_stats(df, item_name='all'):
        tmp = pd.DataFrame()
        if item_name == 'all':
            tmp = df.copy()
        else:
            tmp = df[df['item_name'] == item_name]

        word_pairs = []
        word_pos_dict = defaultdict(lambda: [])
        for idx, row in tmp.iterrows():
            word_pairs.extend(row['comment_seg'])
            for pair in row['comment_seg']:
                word_pos_dict[pair[-1]].append(pair[0])  # 按词性分组

        # group by word
        big_word = list(zip(*word_pairs))[0]
        big_word_dict = Counter(big_word)  # 计算词频
        big_word_list = sorted(big_word_dict.items(), key=lambda x: x[1], reverse=True)
        big_word_list = big_word_list[:50] if item_name == 'all' else big_word_list[:20]  # 选取top50/20

        # group by pos
        word_pos_dict2 = OrderedDict()
        for key, name in config.pos_map.items():
            val = word_pos_dict.get(key, [])
            val_dict = Counter(val)
            val_list = sorted(val_dict.items(), key=lambda x: x[1], reverse=True)
            word_pos_dict2[name] = val_list[:15]  # 选取top15

        return big_word_list, word_pos_dict2


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()
    shop_name = FLAGS.shop_name

    revAnaer = ReviewAnalysis(shop_name)
    revAnaer.dump_stats()

    print('task is done...')