#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 10 May 2019
@author: Nano Zhou
"""

import re
import jieba

def load_stopwords(stopwords_file_path):
    # load stop words
    stop_words_list = []
    with open(stopwords_file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.split()
            stop_words_list.extend(line)
    return set(stop_words_list)

def load_map_file(map_file_path):
    # load word-index map file
    with open(map_file_path, 'r', encoding='utf-8') as fr:
        lines = [line.strip('\n').split('\t') for line in fr]  # both word and index are loaded as str
    word_dict = dict(lines)  # dict: word-index
    return word_dict

def re_seg_comment(comment, stop_words):
    commentSeg = []
    # regularization expression
    try:
        comment = re.sub('\xa0', ',', comment)
        comment = re.sub('[0-9+-@#&_,.:;?!\n\s()""，。：；？！（）、……“”]+', '/', comment)
        comment_split = comment.split('/')  # generate list with sentences
    except:
        print(comment)
    for sentence in comment_split:
        sentenceSeg = jieba.cut(sentence, cut_all=False, HMM=True)
        sentenceSeg_new = []
        for word in sentenceSeg:
            if word not in stop_words:
                sentenceSeg_new.append(word)
        commentSeg.extend(sentenceSeg_new)
    commentSeg = [elem.upper() for elem in commentSeg]  # include English words at first
    return commentSeg
