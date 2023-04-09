#!/usr/bin/env python
#coding=utf-8
"""
Created on Sat, 20 Apr 2019
- do segmentation to generate seg for corpus
- do w2v training based on corpus and save models, params and word.map
@author: Nano Zhou
"""

import config
import nlp_utils.file_path_op as fpo
import data_pipeline

import logging
import numpy as np
import multiprocessing
import pickle
from gensim.models import word2vec
from gensim.models import KeyedVectors
import os
import argparse


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


def gene_seg_corpus(inputs_dir, num_process, task_type='corpus', raw_file_type='csv', save_seg=True):

    # read raw files from input/raw_inputs/corpus
    # gene seg files in input/segments/corpus
    data_pipeline.preproess(inputs_dir, num_process, task_type, raw_file_type, save_seg)
    logging.info('Generate seg files in corpus folder is done...')

def w2v_training(seg_corpus_dir, embedding_size):
    w2v_model_file = 'w2v_embed_' + str(embedding_size) + '.model'
    w2v_vector_file = 'w2v_embed_' + str(embedding_size) + '.txt'
    sentences = word2vec.PathLineSentences(seg_corpus_dir)
    workers = multiprocessing.cpu_count()
    # basic setting in w2v
    w2v_model = word2vec.Word2Vec(sentences=sentences, size=embedding_size, window=5, min_count=5, workers=workers,\
        sg=1, hs=0, negative=10, ns_exponent=0.75, iter=10, sorted_vocab=1)
    w2v_model.save(config.params_dir + w2v_model_file)
    w2v_model.wv.save_word2vec_format(config.params_dir + w2v_vector_file, binary=False)
    logging.info('Word2Vec training is done and data are saved..')

def w2v_map_setup(seg_corpus_dir, map_file, embedding_size):
    w2v_vector_file = 'w2v_embed_' + str(embedding_size) + '.txt'
    vector_file = 'w2v_embed_' + str(embedding_size) + '.pkl'
    word_dict = dict()
    w2v = []
    cnt = 0
    word_dict['<Dummy>'] = cnt
    w2v.append([0.] * embedding_size)  # add dummy vectors at 0-th row
    w2v_vector_file_path = os.path.join(config.params_dir, w2v_vector_file)

    with open(w2v_vector_file_path, 'r', encoding='utf-8') as fr:
        next(fr)  # skip header line
        for line in fr:
            line = line.split()
            assert len(line) == embedding_size + 1, 'A wrong word embedding occurs...'
            cnt += 1
            word_dict[line[0]] = cnt
            w2v.append(line[1:])
    logging.info('Parsing from w2v_embed file is done...')

    w2v = np.asarray(w2v, dtype=np.float32)  # dim = (vocab_size+1) * embedding_size
    vector_file_path = os.path.join(config.params_dir, vector_file)
    pickle.dump(w2v, open(vector_file_path, 'wb'))  # use pickle to save np.array
    logging.info('Embedding vectors is saved as new file...')

    seg_corpus_files = fpo.get_unprocessed_files(seg_corpus_dir, file_type='txt')
    for seg_file_path in seg_corpus_files:
        with open(seg_file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.split()
                for word in line:
                    if word not in word_dict:
                        cnt += 1
                        word_dict[word] = cnt
    sorted_word_dict = sorted(word_dict.items(), key=lambda kv: kv[1])  # sorted by value (key-value), tuple in list
    logging.info('Parsing from seg_corpus files is done...')
    logging.info('Total unique tokens in word_dict is: {}'.format(len(word_dict)-1))
    
    map_file_path = os.path.join(config.params_dir, map_file)
    with open(map_file_path, 'w', encoding='utf-8') as fw:
        for pairs in sorted_word_dict:
            fw.write('{}\t{:d}\n'.format(pairs[0], pairs[1]))  # write as word-index
    logging.info('Map for word-index in training dataset is done...')
    
def w2v_main(inputs_dir, map_file, embedding_size):

    seg_corpus_dir = os.path.join(inputs_dir, 'segments', 'corpus')

    logging.info('Start train w2v...')
    w2v_model_file = 'w2v_embed_' + str(embedding_size) + '.model'
    w2v_vector_file = 'w2v_embed_' + str(embedding_size) + '.txt'
    w2v_training(seg_corpus_dir, embedding_size)

    logging.info('Setup word-index map...')
    vector_file = 'w2v_embed_' + str(embedding_size) + '.pkl'
    w2v_map_setup(seg_corpus_dir, map_file, embedding_size)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_process', 
        type=int,
        default=8,
        help='num of processes in multiprocessing'
    )
    parser.add_argument(
        '--embedding_size', 
        type=int,
        default=100,
        help='embedding size in w2v'
    )
    parser.add_argument(
        '--map_file',
        type=str,
        default='word.map',
        help='map for word-index'
    )

    FLAGS, unparsed = parser.parse_known_args()

    gene_seg_corpus(
        inputs_dir=config.input_data_dir,
        num_process=FLAGS.num_process,
    )

    w2v_main(
        inputs_dir=config.input_data_dir,
        map_file=FLAGS.map_file,
        embedding_size=FLAGS.embedding_size
    )