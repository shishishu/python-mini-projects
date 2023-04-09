#!/usr/bin/env python
#coding=utf-8
# @file  : mlp
# @time  : 5/31/2020 4:37 PM
# @author: shishishu

import os
import tensorflow as tf
from conf import config
from lib.utils.utils import collect_log_key_content
from lib.model.dnnBase import DNNBase
from lib.utils.generalNet import GeneralNet

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'mlp.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8192, 'number of examples per batch')
tf.app.flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'number of training iterations')
tf.app.flags.DEFINE_integer('skip_epoch', 1, 'print intermediate result per skip epoch')
tf.app.flags.DEFINE_integer('skip_step', 10, 'print intermediate result per skip step')

# base class
class MLP(DNNBase):

    def __init__(self, batch_size, learning_rate, l2_reg, num_epoch, skip_epoch, skip_step, fea_size, pred_domain, input_dir, model_type, model_version, hidden_layers, acti_func):
        super(MLP, self).__init__(batch_size, learning_rate, l2_reg, num_epoch, skip_epoch, skip_step, fea_size, pred_domain, input_dir, model_type, model_version)
        self.hidden_layers = hidden_layers
        self.acti_func = acti_func

    def inference(self, *args, **kwargs):
        with tf.name_scope('network'):
            fc = GeneralNet.fully_connected_layers(self.feas, self.hidden_layers, self.acti_func, self.keep_prob, 'fc')
            self.logits = GeneralNet.fully_connected_layer(fc, self.num_class, 'identity', 1.0, 'logits')

def main(_):

    params = DNNBase.default_params()
    params['learning_rate'] = 3e-4
    params['l2_reg'] = 1e-5
    params['model_type'] = 'mlp'
    params['model_version'] = '0020'
    params['input_dir'] = os.path.join(config.DATA_DIR, 'mlp', '002')
    params['hidden_layers'] = [256, 512, 512, 512, 256, 64]
    params['acti_func'] = 'tanh'

    mlper = MLP(**params)
    collect_log_key_content(mlper.log_dict, 'params', params, mlper.log_path)
    logging.info('Start build...')
    mlper.build()
    logging.info('Start run...')
    mlper.run()
    logging.info('Task is done...')


if __name__ == '__main__':

    tf.app.run()



