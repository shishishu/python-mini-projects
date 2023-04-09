#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 12 May 2019
@author: Nano Zhou
"""

import config
import nlp_utils.tf_utils_op as tuo
import nlp_utils.tf_dnn_op as tdo
import nlp_utils.file_path_op as fpo

import logging
import random
import tensorflow as tf
import os
import pickle
import shutil
import json
import time
from datetime import date, timedelta, datetime


# save global log with level = DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=config.log_file_path,
                    filemode='a')

NUM_CLASS = 4  # C
NUM_ONEHOT_LABEL = 80  # 4*20
VOCAB_SIZE = 59529  # number of pertrained words
UNIQUE_TOKEN = 219169  # number of unique token in training dataset

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 100, 'dimension of word embedding')  # E
tf.app.flags.DEFINE_integer('num_lstm_units', 100, 'dimension of hidden units in LSTM')  # H
tf.app.flags.DEFINE_integer('max_sen_len', 240, 'max number of words in one comment')  # M
tf.app.flags.DEFINE_integer('bilstm_depth', 1, 'depth of bilstm network')  # D

tf.app.flags.DEFINE_integer('d_model', 200, 'dim of multi-heads')  # AT*K = 2H
tf.app.flags.DEFINE_integer('num_atten_head', 4, 'num of multi-heads')  # AT
tf.app.flags.DEFINE_boolean('atten_mask', True, 'apply atten_mask before softmax in self-attention')
tf.app.flags.DEFINE_boolean('layer_norm', True, 'apply layer normalization before ffnn')

# {no_pool, mp_seq, mp_seq_topk, mp_unit, mp_unit_head, ap_seq, ap_unit, ap_unit_head}
tf.app.flags.DEFINE_string('pool_method', 'no_pool', 'apply pooling before final softmax')
tf.app.flags.DEFINE_integer('pool_topk', 3, 'pick top k elements in "mp_seq_topk" before final softmax')
tf.app.flags.DEFINE_integer('inter_units', 32, 'inter hidden units before final softmax')  # IU, do not apply when pool_method = 'no_pool'

tf.app.flags.DEFINE_integer('num_thread', 8, 'multi threads')
tf.app.flags.DEFINE_integer('batch_size', 256, 'number of examples per batch')  # B
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'number of training iterations')
tf.app.flags.DEFINE_integer('log_steps', 1, 'save summary every steps')
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'global keep prob')

tf.app.flags.DEFINE_integer('aspect_id', 15, 'id of aspect col for training and validation')  # start from 0, 0-19
tf.app.flags.DEFINE_string('task_type', 'train', 'task type: {train, eval, infer, export}')
tf.app.flags.DEFINE_string('dt_dir', '20190529_1', 'data dt partition')  # change dir based on date
tf.app.flags.DEFINE_boolean('clear_existing_model', False, 'clear existing model or not')


def input_fn(filenames, batch_size, num_epoch, num_thread=1, perform_shuffle=False):
    
    logging.info('Start parsing files: {}'.format(filenames))
    
    dataset = tf.data.TextLineDataset(filenames).map(lambda x: tuo.decode_enc(x, NUM_ONEHOT_LABEL), num_parallel_calls=num_thread).prefetch(1000)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    # hyper-params in models
    embedding_size = params['embedding_size']
    num_lstm_units = params['num_lstm_units']
    max_sen_len = params['max_sen_len']
    bilstm_depth = params['bilstm_depth']
    aspect_id = params['aspect_id']

    d_model = params['d_model']
    num_atten_head = params['num_atten_head']
    atten_mask = params['atten_mask']  # bool
    layer_norm = params['layer_norm']  # bool
    pool_method = params['pool_method']
    pool_topk = params['pool_topk']
    inter_units = params['inter_units']

    assert d_model == 2 * num_lstm_units, 'wrong dim setting in multi-head part'
    
    learning_rate = params['learning_rate']
    # only active dropout when mode == train, others keep_prob = 1.0 always (very important!!!)
    keep_prob = params['keep_prob'] if mode==tf.estimator.ModeKeys.TRAIN else 1.0
    if mode != tf.estimator.ModeKeys.TRAIN:
        assert keep_prob == 1.0, 'use dropout wrongly in eval or infer'

    pretrained_embedding_file = './params/' + 'w2v_embed_' + str(embedding_size) + '.pkl'
    word_embedding = tuo.get_embedding_vectors(pretrained_embedding_file, embedding_size, VOCAB_SIZE, UNIQUE_TOKEN)
    
    batch_sen_len = features['sen_len']
    batch_sen_len = tf.reshape(batch_sen_len, shape=[-1])  # [B], batch_size is dynamic changing
    batch_sen_encode = features['sen_encode']
    batch_sen_encode = tf.reshape(batch_sen_encode, shape=[-1, max_sen_len])  # [B, M]

    with tf.variable_scope('Inputs-Part'):
        inputs = tf.nn.embedding_lookup(word_embedding, batch_sen_encode)  # [B, M, E]
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)  # dropout before send to bilstm

    with tf.variable_scope('Share-RNN-Part'):
        rnn_outputs = tdo.dynamic_deep_bilstm(inputs, batch_sen_len, max_sen_len, num_lstm_units, bilstm_depth, keep_prob)  # [D, B, M, 2H]
        # use last output as default (if need more previous output, use slice to pick targets)
        rnn_outputs = rnn_outputs[-1]  # [B, M, 2H]
    
    with tf.variable_scope('Indep-Atten-Part'):
        multi_header = tdo.MultiHeadAtten(num_atten_head, d_model)
        atten_outputs = multi_header.multi_atten(rnn_outputs, atten_mask)  # [B, M, AT*K]
        atten_outputs = tf.nn.dropout(atten_outputs, keep_prob=keep_prob)  # [B, M, AT*K]

    with tf.variable_scope('Indep-FFNN-Part'):
        indep_inputs = rnn_outputs + atten_outputs  # ResNet, [B, M, 2H], 2H = AT * V
        # No need to differentiate train of infer (the real operations is not clear)
        # Run layer normalization on the last dimension of the tensor (ref to BERT)
        if layer_norm:
            indep_inputs = tf.contrib.layers.layer_norm(inputs=indep_inputs, begin_norm_axis=-1, begin_params_axis=-1, scope='layer_norm')
        ffnn_inputs = tf.nn.dropout(indep_inputs, keep_prob=keep_prob)  # [B, M ,2H]
    
    with tf.variable_scope('Pred-Part'):  ## revise and maxpool case
        
        pools = tdo.PoolSet(ffnn_inputs, max_sen_len, d_model, num_atten_head)
        pool_func = getattr(pools, pool_method)  # work as pools.pool_method
        if pool_method == 'mp_seq_topk':
            pool_outputs = pool_func(pool_topk)  # need additional param 'pool_topk'
            pool_outputs = tf.layers.dense(pool_outputs, units=2*inter_units)  # [B, 2H*topk] -> [B, 2IU]
            pool_outputs = tf.nn.dropout(pool_outputs, keep_prob=keep_prob)
        else:
            pool_outputs = pool_func()
            if pool_method != 'no_pool':  # [B, M*2H]
                if pool_method == ('mp_unit_head' or 'ap_unit_head'):
                    pool_outputs = tf.layers.dense(pool_outputs, units=2*inter_units)  # [B, M*AT] -> [B, 2IU]
                else:
                    pool_outputs = tf.layers.dense(pool_outputs, units=inter_units)  # [B, 2H/M] -> [B, IU]
                pool_outputs = tf.nn.dropout(pool_outputs, keep_prob=keep_prob)
        
        # pool_outputs = tf.reshape(ffnn_inputs, shape=[-1, max_sen_len*2*num_lstm_units])  # [B, M*2H], return to back for dim check
        prob = tdo.softmax_classifier(pool_outputs, NUM_CLASS)  # [B, C]
        pred_cls = tf.argmax(prob, axis=-1)  # [B]

    predictions = {'prob': prob, 'pred_cls': pred_cls}
    # use export_outputs for serving
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    
    # provide an estimator sepc for PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs
        )
    
    # put batch_label after PREDICT!!! Otherwise, it will report issues always
    batch_label = labels[:, (aspect_id * NUM_CLASS) : (aspect_id + 1) * NUM_CLASS]  # [B, C], pick labels in spec aspect

    with tf.variable_scope('Loss-Part'):
        engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch_label, logits=prob)  # [B]
        loss = tf.reduce_mean(engin_loss)  # a scalar, averaged on batch

    true_cls = tf.argmax(batch_label, axis=-1)  # [B]

    eval_metric_ops = dict()
    acc_name = 'accuracy_' + str(aspect_id)
    eval_metric_ops[acc_name] = tf.metrics.accuracy(true_cls, pred_cls, name=acc_name + '_op')  # return 2 items, use last one (update continuously)
    tf.summary.scalar(acc_name, eval_metric_ops[acc_name][1])  # runtime update in training

    # provide an estimator spec for EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )

    # provide an estimator spec for TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
        )
    
def main(_):

    logging.info('\nRunning matt with apsect {}'.format(FLAGS.aspect_id))
    
    if FLAGS.dt_dir == '':
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')  # reuse data from last day if not defined
    ## generate path dynamicly based on model_type
    dirs = fpo.gene_dirs(config.model_dir, config.log_dir, config.output_dir, 'matt', FLAGS.aspect_id, FLAGS.dt_dir)
    model_dir = dirs['model_dir']
    print('model dir is: ', model_dir)
    output_dir = dirs['output_dir']

    data_dir = os.path.join(config.input_data_dir, 'encodes')
    tr_files = fpo.get_unprocessed_files(data_dir + '/train', file_type='txt')
    random.shuffle(tr_files)
    logging.info('Files used in training: {}'.format(tr_files))
    va_files = fpo.get_unprocessed_files(data_dir + '/valid', file_type='txt')
    logging.info('Files used in validation: {}'.format(va_files))
    te_files = fpo.get_unprocessed_files(data_dir + '/test', file_type='txt')
    logging.info('Files used in test: {}'.format(te_files))

    if FLAGS.clear_existing_model:
        try: 
            shutil.rmtree(model_dir)
            time.sleep(0.01)  # time buffer to finish rmtree and make dir again
            os.mkdir(model_dir)
        except Exception as e:
            logging.warning(e, 'at clear_existing_model')
        else:
            logging.warning('existing model is cleaned at {}'.format(model_dir))

    model_params = {
        'embedding_size': FLAGS.embedding_size,
        'num_lstm_units': FLAGS.num_lstm_units,
        'max_sen_len': FLAGS.max_sen_len,
        'bilstm_depth': FLAGS.bilstm_depth,
        'num_atten_head': FLAGS.num_atten_head,
        'd_model': FLAGS.d_model,
        'atten_mask': FLAGS.atten_mask,
        'layer_norm': FLAGS.layer_norm,
        'pool_method': FLAGS.pool_method,
        'pool_topk': FLAGS.pool_topk,
        'inter_units': FLAGS.inter_units,
        'learning_rate': FLAGS.learning_rate,
        'keep_prob': FLAGS.keep_prob,
        'aspect_id': FLAGS.aspect_id
    }

    # save model params with time sequence
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(model_dir + '/params.json', 'a') as fw:
        fw.write(timestamp + '\n')
        json.dump(model_params, fw, indent=4, separators=(',', ':'))
        fw.write('\n\n')

    configs = tf.estimator.RunConfig().replace(log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    MATT = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=model_params,
        config=configs
    )

    if FLAGS.task_type == 'train': 
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, FLAGS.batch_size, FLAGS.num_epoch, FLAGS.num_thread, True),
            max_steps=150000
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, FLAGS.batch_size, 1, FLAGS.num_thread),
            steps=None,
            start_delay_secs=1000,
            throttle_secs=1500
        )
        tf.estimator.train_and_evaluate(MATT, train_spec, eval_spec)
    
    elif FLAGS.task_type == 'eval':
        MATT.evaluate(
            input_fn=lambda: input_fn(va_files, FLAGS.batch_size, num_epoch=1)
        )
    
    elif FLAGS.task_type == 'infer':
        preds = MATT.predict(
            input_fn=lambda: input_fn(te_files, FLAGS.batch_size, num_epoch=1),
            predict_keys='pred_cls'
        )  # generator
        y_pred_cls = []
        for pred in preds:
            y_pred_cls.append(pred['pred_cls'])
        fpo.export_test_prediction(output_dir, y_pred_cls)

    elif FLAGS.task_type == 'export':
        feature_spec = {
            'sen_len': tf.placeholder(dtype=tf.int32, shape=[None], name='sen_len'),
            'sen_encode': tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_sen_len], name='sen_encode')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        server_dir = os.path.join(config.output_dir, 'matt', 'aspect_' + str(FLAGS.aspect_id), 'server', FLAGS.dt_dir)
        if not os.path.exists(server_dir):
            os.makedirs(server_dir)
        MATT.export_savedmodel(server_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()