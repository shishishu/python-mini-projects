#!/usr/bin/env python
#coding=utf-8
# @file  : rnn
# @time  : 6/2/2020 10:04 PM
# @author: shishishu

import tensorflow as tf
from lib.model.dnnBase import DNNBase
from lib.utils.generalNet import GeneralNet
from lib.utils.rnnNet import RnnNet
from lib.utils.utils import collect_log_key_content, collect_log_content
from conf import config
import os

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'rnn.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 2048, 'number of examples per batch')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0, 'l2 regularization')
tf.app.flags.DEFINE_float('group_loss_weight', 0, 'group loss weight')
tf.app.flags.DEFINE_integer('num_epoch', 30, 'number of training iterations')
tf.app.flags.DEFINE_integer('skip_epoch', 1, 'print intermediate result per skip epoch')
tf.app.flags.DEFINE_integer('skip_step', 100, 'print intermediate result per skip step')
tf.app.flags.DEFINE_integer('num_lstm_units', 128, 'number of lstm units')
tf.app.flags.DEFINE_integer('num_deep_rnn', 1, 'number of lstm layers')
tf.app.flags.DEFINE_string('model_version', '0011', 'model version')
tf.app.flags.DEFINE_string('pred_domain', 'gender', 'pred domain')
tf.app.flags.DEFINE_string('hidden_layers', '512,512,64', 'hidden layers')


class RNN(DNNBase):
    def __init__(self, batch_size, learning_rate, l2_reg, num_epoch, skip_epoch, skip_step, fea_size, pred_domain, ad_domains, input_dir, model_type, model_version, embedding_size, onehot_embedding_size, num_lstm_units, num_deep_rnn, hidden_layers, group_loss_weight = 0, acti_func='tanh', train_embedding=False, loss_type='default'):
        super(RNN, self).__init__(batch_size, learning_rate, l2_reg, num_epoch, skip_epoch, skip_step, fea_size, pred_domain, input_dir, model_type, model_version)
        self.embedding_size = embedding_size
        self.onehot_embedding_size = onehot_embedding_size
        self.num_lstm_units = num_lstm_units
        self.num_deep_rnn = num_deep_rnn
        self.hidden_layers = hidden_layers
        self.acti_func = acti_func
        self.train_embedding = train_embedding
        self.ad_domains = ad_domains
        self.num_group = len(config.AGE_BIG_GROUP)  # big group

        dict_data = list(config.AGE_BIG_GROUP_INV.values())
        logging.info('dict data in big group is: {}'.format(dict_data))
        collect_log_content(self.log_dict, 'dict in big group is: {}'.format(config.AGE_BIG_GROUP_INV), self.log_path)
        collect_log_content(self.log_dict, 'dict data in big group is: {}'.format(dict_data), self.log_path)
        self.dict_data = tf.constant(dict_data, dtype=tf.int32)
        self.group_loss_weight = group_loss_weight
        self.loss_type = loss_type

    def inference(self, *args, **kwargs):
        with tf.name_scope('get_embedding'):
            word_embeddings = []
            for ad_domain in self.ad_domains:
                word_embeddings.append(GeneralNet.get_embedding_vectors(self.embedding_size, ad_domain))
        with tf.name_scope('get_inputs'):
            self.inputs = tf.nn.embedding_lookup(word_embeddings[0], self.sen_encode)  # [B, M, E]
            if len(self.ad_domains) > 1:
                self.inputs2 = tf.nn.embedding_lookup(word_embeddings[1], self.sen_encode2)  # [B, M, E]
                self.inputs = tf.concat([self.inputs, self.inputs2], axis=2)  # [B, M, 2E]
                if len(self.ad_domains) > 2:
                    self.inputs3 = tf.nn.embedding_lookup(word_embeddings[2], self.sen_encode3)  # [B, M, E]
                    self.inputs = tf.concat([self.inputs, self.inputs3], axis=2)  # [B, M, 3E]
            self.inputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob)

        with tf.name_scope('network'):
            _, states = RnnNet.dynamic_deep_lstm(self.inputs, self.sen_len, self.num_lstm_units, self.num_deep_rnn)
            self.last_eff_output = states[-1].h  # [B, H], hidden states in last layer
            self.last_eff_output = tf.nn.dropout(self.last_eff_output, keep_prob=self.keep_prob)
            if self.model_type in ['rnn2cate', 'rnn2gender', 'rnn2cross', 'rnn2concat']:
                onehot_output = GeneralNet.fully_connected_layer(self.onehot_encode, self.onehot_embedding_size, self.acti_func, self.keep_prob, 'onehot')
                self.last_eff_output = tf.concat([self.last_eff_output, onehot_output], axis=1)  # [B, H + OHE]
            fc = GeneralNet.fully_connected_layers(self.last_eff_output, self.hidden_layers, self.acti_func, self.keep_prob, 'fc')
            self.logits = GeneralNet.fully_connected_layer(fc, self.num_class, 'identity', 1.0, 'logits')
        if self.pred_domain == 'age' and self.group_loss_weight > 0:
            with tf.name_scope('big_group'):
                self.group_labels = GeneralNet.gene_big_group_labels(self.labels, self.dict_data, self.num_group)
                self.group_logits = GeneralNet.fully_connected_layer(fc, self.num_group, 'identity', 1.0, 'group_logits')

    def loss(self, *args, **kwargs):
        with tf.name_scope('loss'):
            if self.loss_type == 'focal_loss':
                engin_loss = GeneralNet.focal_loss_softmax(labels=self.labels, logits=self.logits, gamma=2.0)
            else:
                engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)  # use logits
            self.loss = tf.reduce_mean(engin_loss)
            if self.l2_reg > 0:
                vars = tf.trainable_variables()
                # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * self.l2_reg
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_reg  # exclude bias
                self.loss += l2_loss
            if self.pred_domain == 'age' and self.group_loss_weight > 0:
                group_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.group_labels, logits=self.group_logits)
                self.loss += self.group_loss_weight * tf.reduce_mean(group_loss)
        return


def main(_):

    params = DNNBase.default_params()
    params['model_type'] = 'rnn2'
    params['model_version'] = FLAGS.model_version
    params['pred_domain'] = FLAGS.pred_domain
    params['ad_domains'] = ['creative_id', 'advertiser_id']
    params['input_dir'] = os.path.join(config.DATA_DIR, 'rnn2', '001')
    params['batch_size'] = FLAGS.batch_size
    params['learning_rate'] = FLAGS.learning_rate
    params['l2_reg'] = FLAGS.l2_reg
    params['group_loss_weight'] = FLAGS.group_loss_weight
    params['num_epoch'] = FLAGS.num_epoch
    params['skip_epoch'] = FLAGS.skip_epoch
    params['skip_step'] = FLAGS.skip_step
    params['embedding_size'] = 64
    params['onehot_embedding_size'] = 32
    params['num_lstm_units'] = FLAGS.num_lstm_units
    params['num_deep_rnn'] = FLAGS.num_deep_rnn
    params['hidden_layers'] = FLAGS.hidden_layers.split(',')  # [256, 512, 512, 512, 256, 64]
    params['acti_func'] = 'tanh'  # tanh
    params['train_embedding'] = False
    params['loss_type'] = 'focal_loss'

    rnner = RNN(**params)
    collect_log_key_content(rnner.log_dict, 'params', params, rnner.log_path)
    logging.info('Start build...')
    rnner.build()
    logging.info('Start run...')
    rnner.run()
    logging.info('Task is done...')


if __name__ == '__main__':

    tf.app.run()