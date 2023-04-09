#!/usr/bin/env python
#coding=utf-8
"""
Created on Sat, 11 May 2019
@author: Nano Zhou
"""

import tensorflow as tf


def dynamic_basic_lstm(inputs, batch_sen_len, num_lstm_units, keep_prob=1.0):
    lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=num_lstm_units, name='lstm_cell')
    # wrapper with dropout in output side
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(lstmCell, output_keep_prob=keep_prob)
    # when time_major = False:
    # inputs, dim = [B, M, E]
    # sen_len, dim = [B]
    # outputs, dim = [B, M, H]
    outputs, state = tf.nn.dynamic_rnn(
        cell=lstmCell,
        inputs=inputs,
        sequence_length=batch_sen_len,
        dtype=tf.float32,
        scope='dynamic_lstm'
    )
    return outputs, state

def softmax_classifier(inputs, num_class):
    # inputs: [B, H]
    num_softmax_input = inputs.get_shape().as_list()[1]  # convert to int
    with tf.variable_scope('softmax'):
        weights = tf.get_variable(
            name='softmax_w',
            shape=[num_softmax_input, num_class],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.get_variable(
            name='softmax_b',
            shape=[num_class],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
    pred = tf.matmul(inputs, weights) + biases  # [B, C]
    prob = tf.nn.softmax(pred, axis=1)  # [B, C]
    return prob

def dynamic_deep_lstm(inputs, batch_sen_len, num_lstm_units, num_deep_rnn, keep_prob=1.0):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_lstm_units, name='lstm_cell')
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells = [cell for _ in range(num_deep_rnn)]  # multi deep layers
    multiCells = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.dynamic_rnn(
        cell=multiCells,
        inputs=inputs,
        sequence_length=batch_sen_len,
        dtype=tf.float32,
        scope='dynamic_deeplstm'
    )
    return outputs, states

def dynamic_bi_lstm(inputs, batch_sen_len, num_lstm_units, keep_prob=1.0, scope='dynamic_bilstm'):
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=num_lstm_units)
    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=num_lstm_units)
    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=batch_sen_len,
        dtype=tf.float32,
        scope=scope
    )
    return (output_fw, output_bw), states

def dynamic_deep_bilstm(inputs, batch_sen_len, max_sen_len, num_lstm_units, bilstm_depth, keep_prob):
    outputs = list()
    input_data = inputs  # [B, M, E] or [B, M, 2H] later
    for i in range(bilstm_depth):
        scope = 'dynamic_bilstm_depth_{}'.format(i)  # start from 0
        (output_fw, output_bw), _ = dynamic_bi_lstm(input_data, batch_sen_len, num_lstm_units, keep_prob, scope)
        tmp_outputs = tf.concat([output_fw, output_bw], axis=-1)  # [B, M, 2H]
        input_data = tmp_outputs
        outputs.append(tmp_outputs)
    outputs = tf.reshape(outputs, shape=[bilstm_depth, -1, max_sen_len, 2*num_lstm_units])  # [D, B, M, 2H]
    return outputs


class MultiHeadAtten:

    def __init__(self, num_atten_head, d_model):
        assert d_model % num_atten_head == 0, 'wrong setting in num of attention head'
        self.d_model = d_model
        self.head = num_atten_head  # AT
        self.d_k = d_model // num_atten_head  # K, # assume d_v = d_k always
        self.p_atten = None
        
    def multi_atten(self, inputs, mask=True):
        # [B, M, 2H], d_model = 2H
        batch_size = tf.shape(inputs)[0]
        qkv = list(map(lambda x: tf.layers.dense(x, units=self.d_model), (inputs, inputs, inputs)))  # [B, M, d_model]
        # [B, M, d_model] -> reshape: [B, M, AT, K] -> transpose: [B, AT, M, K]
        query, key, value = list(map(lambda x: tf.transpose(tf.reshape(x, shape=[batch_size, -1, self.head, self.d_k]), perm=[0, 2, 1, 3]), qkv))
        atten_outputs, self.p_atten = MultiHeadAtten.atten_net(query, key, value, mask)
        # [B, AT, M, K] -> transpose: [B, M, AT, K] -> reshape: [B, M, AT*K]
        atten_outputs = tf.reshape(tf.transpose(atten_outputs, perm=[0, 2, 1, 3]), shape=[batch_size, -1, self.head * self.d_k])
        return atten_outputs

    @staticmethod
    def atten_net(query, key, value, mask):
        d_k = tf.shape(query)[-1]  # q, k, v: [B, AT, M, K]
        scale = tf.sqrt(tf.cast(d_k, tf.float32))  # a scalar
        # [B, AT, M, K] * [B, AT, K, M] = [B, AT, M, M]
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))/scale
        if mask:
            scores = MultiHeadAtten.atten_mask(scores)  # replace 0 with -1e9
        p_atten = tf.nn.softmax(scores, axis=-1)  # [B, AT, M, M]
        return tf.matmul(p_atten, value), p_atten  # output: [B, AT, M, K]

    @staticmethod
    def atten_mask(scores, force_val=-1e9):
        masker = tf.equal(scores, 0)
        forcer = tf.multiply(tf.ones_like(scores, dtype=tf.float32), force_val)
        return tf.where(masker, forcer, scores)


class PoolSet:

    def __init__(self, inputs, max_sen_len, d_model, num_atten_head):
        self.inputs = inputs  # [B, M, 2H=AT*K]
        self.max_sen_len = max_sen_len
        self.d_model = d_model
        self.num_atten_head = num_atten_head
        self.d_k = d_model // num_atten_head

    def no_pool(self):
        return tf.reshape(self.inputs, shape=[-1, self.max_sen_len * self.d_model])  # [B, M*2H]

    def mp_seq(self):
        return tf.reduce_max(self.inputs, axis=1)  # [B, 2H]
    
    def mp_seq_topk(self, topk):
        inputs_trans = tf.transpose(self.inputs, perm=[0, 2, 1])  # [B, 2H, M]
        top_result = tf.nn.top_k(inputs_trans, k=topk).values  # [B, 2H, topk]
        return tf.reshape(top_result, shape=[-1, self.d_model * topk])  # [B, 2H*topk]

    def mp_unit(self):
        return tf.reduce_max(self.inputs, axis=-1)  # [B, M]
    
    def mp_unit_head(self):
        inputs_trans = tf.reshape(self.inputs, shape=[-1, self.max_sen_len, self.num_atten_head, self.d_k])  # [B, M, AT, K]
        inputs_mp = tf.reduce_max(inputs_trans, axis=-1)  # [B, M, AT]
        return tf.reshape(inputs_mp, shape=[-1, self.max_sen_len * self.num_atten_head])  # [B, M*AT]

    def ap_seq(self):
        return tf.reduce_mean(self.inputs, axis=1)  # [B, 2H]
    
    def ap_unit(self):
        return tf.reduce_mean(self.inputs, axis=-1)  # [B, M]
    
    def ap_unit_head(self):
        inputs_trans = tf.reshape(self.inputs, shape=[-1, self.max_sen_len, self.num_atten_head, self.d_k])  # [B, M, AT, K]
        inputs_ap = tf.reduce_mean(inputs_trans, axis=-1)  # [B, M, AT]
        return tf.reshape(inputs_ap, shape=[-1, self.max_sen_len * self.num_atten_head])  # [B, M*AT]
