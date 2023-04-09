#!/usr/bin/env python
#coding=utf-8
"""
Created on Sat, 20 Apr 2019
- use Pool for multiprocessing, may be different in window (current) and linux system!
- when task_type = 'corpus', only do segmentation for w2v training
- for other task types, do 'enc' to finish entire data pipeline
@author: Nano Zhou
"""

import config
# import utils
import nlp_utils.file_path_op as fpo
import nlp_utils.re_seg_op as rso

import logging
import jieba
import csv
import os
import sys
import multiprocessing
import argparse
import ast


VOCAB_SIZE = 59529  # number of pertrained words
MAX_SEN_LEN = 240  # sen_lengths setting in dynamic rnn
NUM_CLASS = 4
NUM_ASPECT = 20

# convert variables to str to enable write them into files directly

# save global log with level = DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=config.log_file_path,
                    filemode='a')
# output runtime log with level = INFO
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def pool_initializer():
    global stop_words  # it will be used in re_seg_comment()
    stop_words = rso.load_stopwords(config.dict_dir + 'stopwords.txt')
    global yhot_mapping  # it will be used in changey_to_onehot()
    yhot_mapping = {'-2': 0, '-1': 1, '0': 2, '1': 3}
    global word_dict  # it will be used in encode_seg_comment()
    word_dict = rso.load_map_file(config.params_dir + 'word.map')
    global UNIQUE_TOKEN
    UNIQUE_TOKEN = len(word_dict) - 1  # calculate UNIQUE_TOKEN dynamicly (based on word_dict)
    global max_sen_len
    max_sen_len = MAX_SEN_LEN

def pool_initializer_corpus():
    global stop_words  # it will be used in re_seg_comment()
    stop_words = rso.load_stopwords(config.dict_dir + 'stopwords.txt')

def split_seg_line(line):
    comment = line[1]
    labels = line[2:]
    commentSeg = rso.re_seg_comment(comment, stop_words)
    return labels, commentSeg

def changey_to_onehot(labels):
    labels_onehot = []
    for label in labels:
        tmp = [str(0)] * NUM_CLASS
        tmp[yhot_mapping[label]] = str(1)
        labels_onehot.extend(tmp)
    return labels_onehot

def encode_seg_comment(comment):
    # assign UNIQUE_TOKEN+1 as new index for unseen words <UNK>
    word_idx = [word_dict.get(word, str(UNIQUE_TOKEN+1)) for word in comment]
    if len(word_idx) <= max_sen_len:
        sen_len = len(word_idx)
        word_idx = word_idx + [str(0)] * (max_sen_len - len(word_idx))  # padding with 0
    else:
        sen_len = max_sen_len
        word_idx = word_idx[:max_sen_len]
    return str(sen_len), word_idx

def preproess(inputs_dir, num_process, task_type='train', raw_file_type='csv', save_seg=True):

    raw_inputs_dir = os.path.join(inputs_dir, 'raw_inputs', task_type)

    unprocessed_files = fpo.get_unprocessed_files(raw_inputs_dir, file_type=raw_file_type)
    num_unprocessed_files = len(unprocessed_files)
    if num_unprocessed_files:
        logging.info('Total number of files pending to process is: {}'.format(num_unprocessed_files))
    else:
        logging.info('No new files pending to process')
        sys.exit()
    
    # load user dict
    jieba.load_userdict(config.dict_dir + 'userdict.txt')

    # use multiprocess with Pool (time cost is low but memory cost is higher)
    if task_type == 'corpus': 
        pool = multiprocessing.Pool(processes=num_process, initializer=pool_initializer_corpus)
        save_seg = True  # must save seg if task type is 'corpus'
    else:
        pool = multiprocessing.Pool(processes=num_process, initializer=pool_initializer)

    for raw_file_path in unprocessed_files:
        
        with open(raw_file_path, 'r', encoding='utf-8') as fr:
            reader = csv.reader(fr)
            next(reader)  # skip header line
            split_results = pool.map(split_seg_line, [line for line in reader])
            total_labels, seg_comments = zip(*split_results)
            logging.info('Loading and segmentation is done...')

        _, raw_name_stem, _ = fpo.get_dir_file_name(raw_file_path)  # 'raw.csv'
        fpo.rename_processed_file(raw_file_path)  # 'raw_1.csv'

        if save_seg:
            seg_file_name = raw_name_stem + '_seg.txt'  # 'raw_seg.txt'
            seg_file_path = os.path.join(config.input_data_dir, 'segments', task_type, seg_file_name)
            with open(seg_file_path, 'w', encoding='utf-8') as fw:
                for seg_comment in seg_comments:
                    fw.write(' '.join(seg_comment))
                    fw.write('\n')
            logging.info('Segmentation files is saved...')

        if task_type == 'corpus':
            logging.info('No need to execute enc step, stop and continue to seg...')  # do segmentation only for corpus collection
            continue
        else:
            logging.info('Continue to obtain enc file...')
    
        # all the labels are converted to str
        if task_type == 'test':
            # fill 0 to make structure unchanged
            total_labels_onehot = [[str(0) for _ in range(NUM_ASPECT * NUM_CLASS)] for _ in range(len(total_labels))]
        else:
            total_labels_onehot = pool.map(changey_to_onehot, total_labels)  # shape of sub-list: [20] -> [80]
        
        encode_results = pool.map(encode_seg_comment, seg_comments)
        sen_len, sen_encode = zip(*encode_results)

        sen_record = zip(total_labels_onehot, sen_len, sen_encode)

        enc_file_name = raw_name_stem + '_enc.txt'  # 'raw_enc.txt'
        enc_file_path = os.path.join(config.input_data_dir, 'encodes', task_type, enc_file_name)
        with open(enc_file_path, 'w') as fw:
            for pairs in sen_record:
                fw.write(' '.join(pairs[0]) + ' ' + pairs[1] + ' ' + ' '.join(pairs[2]))
                fw.write('\n')
        logging.info('Encodes file is saved...')

    pool.close()
    pool.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_process',
        type=int,
        default=8,
        help='num of processes in multiprocessing'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='train',
        help='task type, candidates are train, valid, test, corpus'
    )
    parser.add_argument(
        '--save_seg',
        type=ast.literal_eval,
        default=True,
        help='save intermediate seg comments as new file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    preproess(
        inputs_dir=config.input_data_dir,
        num_process=FLAGS.num_process,
        task_type=FLAGS.task_type,
        raw_file_type='csv',
        save_seg=FLAGS.save_seg
    )