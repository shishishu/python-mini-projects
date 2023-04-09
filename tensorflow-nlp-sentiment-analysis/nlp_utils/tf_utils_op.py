#!/usr/bin/env python
#coding=utf-8
"""
Created on Sat, 11 May 2019
@author: Nano Zhou
"""

import tensorflow as tf
import pickle


def get_embedding_vectors(embedding_file_path, embedding_size, VOCAB_SIZE, UNIQUE_TOKEN):

    with tf.variable_scope('embeddings'):
        # use pretrained embedding for frequent words, dim = (vocab_size+1) * embedding_size (dummy at 0th row)
        pretrained_embedding = pickle.load(open(embedding_file_path, 'rb'))
        pretrained_embedding = tf.get_variable(
            name='pretrained',
            shape=pretrained_embedding.shape,
            initializer=tf.constant_initializer(pretrained_embedding, dtype=tf.float32), 
            trainable=False
        )
        # random init for infrequent words appear in training dataset, dim = (unique_token-vocab_size) * embedding_size
        num_other_train = UNIQUE_TOKEN - VOCAB_SIZE
        other_train_embedding = tf.get_variable(
            name='other_train',
            shape=[num_other_train, embedding_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=False
        )
        # random init for unkown words, dim = 1 * embedding_size
        unknown_embedding = tf.get_variable(
            name='unknown',
            shape=[1, embedding_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=False
        )
        # concat all for final word embedding, dim = (unique_token+2) * embedding_size (one for dummy at 0th, another for unk at end)
    word_embedding = tf.concat([pretrained_embedding, other_train_embedding, unknown_embedding], axis=0)
    return word_embedding

def decode_enc(line, NUM_ONEHOT_LABEL):  # decode line by line
    columns = tf.string_split([line], delimiter=' ')  # pack line into tensor
    labels = tf.string_to_number(columns.values[0: NUM_ONEHOT_LABEL], out_type=tf.float32)
    sen_len = tf.string_to_number(columns.values[NUM_ONEHOT_LABEL], out_type=tf.int32)
    sen_encode = tf.string_to_number(columns.values[NUM_ONEHOT_LABEL + 1:], out_type=tf.int32)
    return {'sen_len': sen_len, 'sen_encode': sen_encode}, labels

def add_text_summary(params):
    text_summary = []
    with tf.name_scope('text'):
        for key, val in params.items():
            tmp_summary = tf.summary.text(key, tf.convert_to_tensor(key + ': {}'.format(val)))
            text_summary.append(tmp_summary)
    text_summary_op = tf.summary.merge(text_summary)
    return text_summary_op

def aspect_embedding_tile(aspect_embedding, batch_sen_len, max_sen_len):
    aspect_embedding = tf.reshape(aspect_embedding, shape=[1, 1, -1])  # [1, 1, AE]
    batch_size = tf.shape(batch_sen_len)[0]  # could not use self.batch_size directly as it could change dynamicly
    aspect_embedding_extend = tf.tile(aspect_embedding, multiples=[batch_size, max_sen_len, 1])  # [B, M, AE]
    mask = tf.sequence_mask(batch_sen_len, maxlen=max_sen_len)  # [B, M]
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)  # [B, M, 1]
    aspect_embedding_extend = tf.multiply(aspect_embedding_extend, mask) # [B, M, AE]
    return aspect_embedding_extend

def add_atten_vectors(num_lstm_units, aspect_embedding_size):
    # new added aspect embedding at input/output side: trainable=True, no regularization
    with tf.variable_scope('aspect_embedding'):
        aspect_embedding_input = tf.get_variable(
            name='aspect_embedding_input',
            shape=[aspect_embedding_size],
            initializer=tf.random_uniform_initializer(-1.0, 1.0)  # same level as frequent words after w2v (-1,1)
        )
        aspect_embedding_output = tf.get_variable(
            name='aspect_embedding_output',
            shape=[aspect_embedding_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1)  # smaller than input side as it will concat with hidden units
        )

    with tf.variable_scope('projection'):
        weightsH = tf.get_variable(
            name='projection_Wh',
            shape=[num_lstm_units, num_lstm_units],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        weightsA = tf.get_variable(
            name='projection_Wa',
            shape=[aspect_embedding_size, aspect_embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        weightsP = tf.get_variable(
            name='projection_Wp',
            shape=[num_lstm_units, num_lstm_units],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        weightsX = tf.get_variable(
            name='projection_Wx',
            shape=[num_lstm_units, num_lstm_units],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
    with tf.variable_scope('attention'):
        weightsAtten = tf.get_variable(
            name='attention_alpha',
            shape=[num_lstm_units + aspect_embedding_size],  # [H+AE]
            initializer=tf.random_uniform_initializer(-0.1, 0.1)
        )
    atten_vectors = {
        'aspect_embedding': [aspect_embedding_input, aspect_embedding_output], 
        'projection': [weightsH, weightsA, weightsP, weightsX],
        'attention': weightsAtten
    }
    return atten_vectors

def concat_word_aspect(outputs_word, outputs_aspect, atten_vectors, num_lstm_units, aspect_embedding_size, max_sen_len):
    tmp_outputs_word = tf.reshape(outputs_word, shape=[-1, num_lstm_units])  # [B*M, H]
    weightsH = atten_vectors['projection'][0]  # [H, H]
    # [B*M, H] * [H, H] = [B*M, H] -> [B, M, H]
    tmp_outputs_word = tf.reshape(tf.matmul(tmp_outputs_word, weightsH), shape=[-1, max_sen_len, num_lstm_units])
    tmp_outputs_aspect = tf.reshape(outputs_aspect, shape=[-1, aspect_embedding_size])  # [B*M, AE]
    weightsA = atten_vectors['projection'][1]  # [AE, AE]
    # [B*M, AE] * [AE, AE] = [B*M, AE] -> [B, M, AE]
    tmp_outputs_aspect = tf.reshape(tf.matmul(tmp_outputs_aspect, weightsA), shape=[-1, max_sen_len, aspect_embedding_size])
    return tf.concat([tmp_outputs_word, tmp_outputs_aspect], axis=-1)  # [B, M, H+AE]

def self_atten_init(num_lstm_units, num_atten_head, atten_key_dim, atten_val_dim):
    W_q = tf.get_variable(
        name='WQ_weights',
        shape=[num_atten_head, 2*num_lstm_units, atten_key_dim],  # [AT, 2H, K]
        initializer=tf.random_normal_initializer(mean=0, stddev=0.1)
    )
    W_k = tf.get_variable(
        name='WK_weights',
        shape=[num_atten_head, 2*num_lstm_units, atten_key_dim],  # [AT, 2H, K]
        initializer=tf.random_normal_initializer(mean=0, stddev=0.1)
    )
    W_v = tf.get_variable(
        name='WV_weights',
        shape=[num_atten_head, 2*num_lstm_units, atten_val_dim],  # [AT, 2H, V]
        initializer=tf.random_normal_initializer(mean=0, stddev=0.1)
    )
    return (W_q, W_k, W_v)