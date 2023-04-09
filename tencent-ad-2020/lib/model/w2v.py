#!/usr/bin/env python
#coding=utf-8
# @file  : w2v
# @time  : 5/17/2020 9:08 PM
# @author: shishishu

import logging
import multiprocessing
from gensim.models import word2vec
import os
import argparse
import pickle
import glob
import numpy as np
from conf import config
from lib.utils.fileOperation import FileOperation

# save global log with level = DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'w2v.log'),
                    filemode='a')
# output runtime log with level = INFO
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=64, help='embedding size in w2v')
parser.add_argument('--num_process', type=int, default=10, help='num of processes in multiprocessing')


class Word2Vec:

    def __init__(self, corpus_dir, embedding_size, num_process, ad_domain):
        self.embedding_size = embedding_size
        self.num_process = num_process
        self.ad_domain = ad_domain
        self.corpus_dir = os.path.join(corpus_dir, self.ad_domain)
        self.model_dir = os.path.join(config.MODEL_DIR, 'w2v_' + str(self.embedding_size) + '_' + self.ad_domain)
        FileOperation.safe_mkdir(self.model_dir)

    def w2v_train(self):
        w2v_model_file = 'w2v_embed_' + str(self.embedding_size) + '.model'
        w2v_vector_file = 'w2v_embed_' + str(self.embedding_size) + '.txt'
        sentences = word2vec.PathLineSentences(self.corpus_dir)
        # workers = multiprocessing.cpu_count()
        # basic setting in w2v
        w2v_model = word2vec.Word2Vec(
            sentences=sentences,
            size=self.embedding_size,
            window=5,
            min_count=5,
            workers=self.num_process,
            sg=1,
            hs=0,
            negative=10,
            ns_exponent=0.75,
            iter=10,
            sorted_vocab=1
        )
        w2v_model.save(os.path.join(self.model_dir, w2v_model_file))
        w2v_model.wv.save_word2vec_format(os.path.join(self.model_dir, w2v_vector_file), binary=False)
        logging.info('Word2Vec training is done and data are saved..')

    def w2v_map_setup(self):
        w2v_vector_file = 'w2v_embed_' + str(self.embedding_size) + '.txt'
        vector_file = 'w2v_embed_' + str(self.embedding_size) + '.pkl'
        word_dict = dict()
        word_dict['<Dummy>'] = 0
        word_dict['CLS'] = 1
        word_dict['UNK'] = 2

        w2v = []
        cnt = len(word_dict) - 1
        with open(os.path.join(self.model_dir, w2v_vector_file), 'r', encoding='utf-8') as fr:
            next(fr)  # skip header line
            for line in fr:
                line = line.split()
                assert len(line) == self.embedding_size + 1, 'A wrong word embedding occurs...'
                cnt += 1
                word_dict[line[0]] = cnt
                w2v.append(line[1:])
        logging.info('Parsing from w2v_embed file is done...')

        w2v = np.asarray(w2v, dtype=np.float32)
        vector_file_path = os.path.join(self.model_dir, vector_file)
        pickle.dump(w2v, open(vector_file_path, 'wb'))  # use pickle to save np.array
        logging.info('Embedding vectors is saved as new file...')

        sorted_word_dict = sorted(word_dict.items(), key=lambda kv: kv[1])
        with open(os.path.join(self.model_dir, 'word.map'), 'w', encoding='utf-8') as fw:
            for pairs in sorted_word_dict:
                fw.write('{}\t{:d}\n'.format(pairs[0], pairs[1]))  # write as word-index
        logging.info('Map for word-index in training dataset is done...')
        return


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    w2ver = Word2Vec(
        corpus_dir=os.path.join(config.DATA_DIR, 'base_0611', 'corpus'),
        embedding_size=FLAGS.embedding_size,
        num_process=min(FLAGS.num_process, multiprocessing.cpu_count()),
        ad_domain='product_id'
    )
    w2ver.w2v_train()
    w2ver.w2v_map_setup()