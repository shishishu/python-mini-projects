#!/usr/bin/env python
#coding=utf-8
# @file  : rnnNet
# @time  : 6/2/2020 10:15 PM
# @author: shishishu

import tensorflow as tf


class RnnNet:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def dynamic_deep_bilstm(inputs, batch_sen_len, max_sen_len, num_lstm_units, bilstm_depth, keep_prob):
        outputs = list()
        input_data = inputs  # [B, M, E] or [B, M, 2H] later
        for i in range(bilstm_depth):
            scope = 'dynamic_bilstm_depth_{}'.format(i)  # start from 0
            (output_fw, output_bw), _ = RnnNet.dynamic_bi_lstm(input_data, batch_sen_len, num_lstm_units, keep_prob, scope)
            tmp_outputs = tf.concat([output_fw, output_bw], axis=-1)  # [B, M, 2H]
            input_data = tmp_outputs
            outputs.append(tmp_outputs)
        outputs = tf.reshape(outputs, shape=[bilstm_depth, -1, max_sen_len, 2 * num_lstm_units])  # [D, B, M, 2H]
        return outputs