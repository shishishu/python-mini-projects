#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 10 May 2019
@author: Nano Zhou
- Basic LSTM and variants, including Deep-LSTM and Bi-LSTM, atten-LSTM
"""

import config
import nlp_utils.file_path_op as fpo
import nlp_utils.tf_utils_op as tuo
import nlp_utils.tf_dnn_op as tdo
import nlp_utils.plot_img_op as pio

import numpy as np
import tensorflow as tf
import logging
import random
import os
from sklearn import metrics
from datetime import datetime

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

VOCAB_SIZE = 59529  # number of pertrained words
UNIQUE_TOKEN = 219169  # number of unique token in training dataset
NUM_CLASS = 4  # C
NUM_ONEHOT_LABEL = 80  # 4*20

# update params after tunning
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_size', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('num_lstm_units', 100, 'dimension of hidden units in LSTM')
tf.app.flags.DEFINE_integer('max_sen_len', 240, 'max number of words in one comment')
tf.app.flags.DEFINE_integer('batch_size', 500, 'number of examples per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'number of training iterations')
tf.app.flags.DEFINE_integer('skip_step', 50, 'step interval of intermediate output')
tf.app.flags.DEFINE_integer('skip_epoch', 1, 'epoch interval of intermediate image of whole output in validation')
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'keep_prob in drop out')
tf.app.flags.DEFINE_integer('num_deep_rnn', 2, 'layers of MultiRNNCell')
tf.app.flags.DEFINE_integer('aspect_embedding_size', 100, 'dimension of aspect embedding')

tf.app.flags.DEFINE_integer('aspect_id', 15, 'id of aspect col for training and validation')  # start from 0, 0-19
tf.app.flags.DEFINE_string('model_type', 'lstm', 'type of LSTM model')  ## default=lstm, candidates are 'lstm', 'bilstm', 'deeplstm', 'aeatlstm'


class LSTM:

    def __init__(self, embedding_size, num_lstm_units, max_sen_len, batch_size, learning_rate, num_epoch, aspect_id, model_type):
        self.embedding_size = embedding_size
        self.num_lstm_units = num_lstm_units
        self.max_sen_len = max_sen_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.aspect_id = aspect_id
        self.model_type = model_type  ## use for model selection
        self.keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob')

        self.params = {
            'aspect_id': self.aspect_id,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'keep_prob': FLAGS.keep_prob,
            'embedding_size': self.embedding_size,
            'num_lstm_units': self.num_lstm_units,
            'num_epoch': self.num_epoch,
            'model_type': self.model_type
        }

        pretrained_embedding_file = './params/' + 'w2v_embed_' + str(self.embedding_size) + '.pkl'
        self.word_embedding = tuo.get_embedding_vectors(pretrained_embedding_file, self.embedding_size, VOCAB_SIZE, UNIQUE_TOKEN)

    def get_inputs(self, batch_sen_encode, batch_sen_len):
        inputs_word = tf.nn.embedding_lookup(self.word_embedding, batch_sen_encode)  # [B, M, E]
        return tf.nn.dropout(inputs_word, keep_prob=self.keep_prob)
    
    def get_outputs(self, inputs, batch_sen_len):
        logging.info('dynamic_run in lstm is calling...')
        _, state = tdo.dynamic_basic_lstm(inputs, batch_sen_len, self.num_lstm_units)
        last_eff_output = state.h  # [B, H]
        return tf.nn.dropout(last_eff_output, keep_prob=self.keep_prob)
    
    def text_summary_op(self):
        return tuo.add_text_summary(self.params)

    def run(self):
        logging.info('Model type: {}'.format(self.model_type))

        data_dir = os.path.join(config.input_data_dir, 'encodes')
        tr_files = fpo.get_unprocessed_files(data_dir + '/train', file_type='txt')
        random.shuffle(tr_files)
        logging.info('Files used in training: {}'.format(tr_files))
        va_files = fpo.get_unprocessed_files(data_dir + '/valid', file_type='txt')
        logging.info('Files used in validation: {}'.format(va_files))
        te_files = fpo.get_unprocessed_files(data_dir + '/test', file_type='txt')
        logging.info('Files used in test: {}'.format(te_files))

        tr_data = tf.data.TextLineDataset(tr_files).map(lambda x: tuo.decode_enc(x, NUM_ONEHOT_LABEL)).prefetch(10*self.batch_size)
        tr_data = tr_data.shuffle(buffer_size=10*self.batch_size).batch(self.batch_size)
        va_data = tf.data.TextLineDataset(va_files).map(lambda x: tuo.decode_enc(x, NUM_ONEHOT_LABEL))
        va_data = va_data.batch(self.batch_size)
        te_data = tf.data.TextLineDataset(te_files).map(lambda x: tuo.decode_enc(x, NUM_ONEHOT_LABEL))
        te_data = te_data.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
        batch_features, batch_labels = iterator.get_next()
        tr_init = iterator.make_initializer(tr_data, name='tr_init')
        va_init = iterator.make_initializer(va_data, name='va_init')
        te_init = iterator.make_initializer(te_data, name='te_init')

        batch_label = batch_labels[:, (self.aspect_id * NUM_CLASS) : (self.aspect_id + 1) * NUM_CLASS]  # [B, C], pick labels in spec aspect
        batch_sen_len = tf.reshape(batch_features['sen_len'], shape=[-1])  # [B]
        batch_sen_encode = tf.reshape(batch_features['sen_encode'], shape=[-1, self.max_sen_len])  # [B, M]

        inputs = self.get_inputs(batch_sen_encode, batch_sen_len)  # [B, M, E]
        outputs = self.get_outputs(inputs, batch_sen_len)  # [B, H]
        prob = tdo.softmax_classifier(outputs, NUM_CLASS)  # [B, C]

        with tf.name_scope('loss'):
            engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch_label, logits=prob)  # [B]
            loss = tf.reduce_mean(engin_loss)
        
        with tf.name_scope('train'):
            gstep = tf.Variable(0, trainable=False, name='global_step')
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=gstep)
        
        with tf.name_scope('predict'):
            y_true = tf.argmax(batch_label, axis=1)  # transfer to [0,1,2,3], dim = batch_size
            y_pred = tf.argmax(prob, axis=1)  # pick index of max prob, dim = batch_size
        
        with tf.name_scope('batch_based'):
            accuracy = tf.placeholder(tf.float32, shape=None, name='batch_accuracy')
            f1_score = tf.placeholder(tf.float32, shape=None, name='batch_f1_score')
            loss_summary = tf.summary.scalar('batch_loss', loss)  # update dependently later
            accuracy_summary = tf.summary.scalar('batch_accuracy', accuracy)
            f1_score_summary = tf.summary.scalar('batch_f1_score', f1_score)

        batch_summary_op = tf.summary.merge([accuracy_summary, f1_score_summary])

        with tf.name_scope('epoch_based'):  # summary op after one epoch
            epoch_loss = tf.placeholder(tf.float32, shape=None, name='epoch_loss')
            epoch_accuracy = tf.placeholder(tf.float32, shape=None, name='epoch_accuracy')
            epoch_f1_score = tf.placeholder(tf.float32, shape=None, name='epoch_f1_score')
            epoch_loss_summary = tf.summary.scalar('epoch_loss', epoch_loss)
            epoch_accuracy_summary = tf.summary.scalar('epoch_accuracy', epoch_accuracy)
            epoch_f1_score_summary = tf.summary.scalar('epoch_f1_score', epoch_f1_score)
        
        epoch_summary_op = tf.summary.merge([epoch_loss_summary, epoch_accuracy_summary, epoch_f1_score_summary])
        
        text_summary_op = self.text_summary_op()  # may change in different models

        with tf.Session() as sess:
            # create folder based on timestamp
            timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

            ## generate path dynamicly based on model_type
            dirs = fpo.gene_dirs(config.model_dir, config.log_dir, config.output_dir, self.model_type, self.aspect_id, timestamp)
            model_dir = dirs['model_dir']
            output_dir = dirs['output_dir']
            runlog_dir = dirs['runlog_dir']
            train_log_dir = dirs['train_log_dir']
            valid_log_dir = dirs['valid_log_dir']

            # record key params in one txt for quick retrieval
            retrieval_file_path = os.path.join(runlog_dir, 'runlog.txt')
            fpo.save_retrival_file(retrieval_file_path, timestamp, self.params)

            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            valid_summary_writer = tf.summary.FileWriter(valid_log_dir)  # no sess.graph in valid

            sess.run(tf.global_variables_initializer())

            train_summary_writer.add_summary(sess.run(text_summary_op))  # add text in the summary (key params)

            saver = tf.train.Saver()
            # restore model, but ignore at first
            ckpt = tf.train.get_checkpoint_state(model_dir, latest_filename='checkpoint')
            if ckpt and ckpt.model_checkpoint_state:
                saver.restore(sess, ckpt.model_checkpoint_state)  # in 'checkpoint' file, model_checkpoint_state: 'lstm-1000'
                logging.info('Loading saved model...{}'.format(ckpt.model_checkpoint_state))
            else:
                logging.info('New model is created...')

            step = gstep.eval()

            for i in range(1, self.num_epoch+1):
                # train on tr_data
                logging.info('Start training at epoch: {}'.format(i))
                sess.run(tr_init)
                total_loss = 0
                display_loss = 0
                num_batch = 0
                train_y_true = []
                train_y_pred = []
                try:
                    while True:
                        _, loss_, y_true_, y_pred_, loss_summary_ = sess.run(
                            [train_op, loss, y_true, y_pred, loss_summary], feed_dict={self.keep_prob: FLAGS.keep_prob}
                        )

                        train_summary_writer.add_summary(loss_summary_, global_step=step)

                        y_true_ = y_true_.tolist()
                        y_pred_ = y_pred_.tolist()
                        train_y_true.extend(y_true_)
                        train_y_pred.extend(y_pred_)

                        batch_summary = sess.run(
                            batch_summary_op,
                            feed_dict={accuracy: metrics.accuracy_score(y_true_, y_pred_), f1_score: metrics.f1_score(y_true_, y_pred_, average='macro')}
                        )
                        train_summary_writer.add_summary(batch_summary, global_step=step)

                        total_loss += loss_
                        display_loss += loss_
                        if (step + 1) % FLAGS.skip_step == 0:
                            avg_loss = display_loss / FLAGS.skip_step
                            display_loss = 0
                            logging.info('Training loss at last {0} steps: {1:.8f}'.format(FLAGS.skip_step, avg_loss))
                        if (step + 1) % (5 * FLAGS.skip_step) == 0:
                            saver.save(sess, os.path.join(model_dir, self.model_type), global_step=(step+1))  # prefix='model/lstm-...'
                        step += 1
                        num_batch += 1
                except tf.errors.OutOfRangeError:
                    pass

                train_epoch_summary = sess.run(
                    epoch_summary_op,
                    feed_dict={epoch_loss: total_loss/num_batch, epoch_accuracy: metrics.accuracy_score(train_y_true, train_y_pred),
                    epoch_f1_score: metrics.f1_score(train_y_true, train_y_pred, average='macro')}
                )
                train_summary_writer.add_summary(train_epoch_summary, global_step=i)  # record as whole epoch
                logging.info('Average training loss at epoch {0}: {1:.8f}'.format(i, total_loss/num_batch))

                # eval on va_data
                logging.info('Start eval on va_data at epoch: {}'.format(i))
                sess.run(va_init)
                valid_total_loss = 0
                valid_num_batch = 0
                valid_y_true = []
                valid_y_pred = []
                try:
                    while True:
                        loss_, y_true_, y_pred_ = sess.run([loss, y_true, y_pred], feed_dict={self.keep_prob: 1.0})
                        y_true_ = y_true_.tolist()
                        y_pred_ = y_pred_.tolist()
                        valid_y_true.extend(y_true_)
                        valid_y_pred.extend(y_pred_)
                        valid_total_loss += loss_
                        valid_num_batch += 1
                except tf.errors.OutOfRangeError:
                    pass
                valid_epoch_summary = sess.run(
                    epoch_summary_op,
                    feed_dict={epoch_loss: valid_total_loss/valid_num_batch, epoch_accuracy: metrics.accuracy_score(valid_y_true, valid_y_pred),
                    epoch_f1_score: metrics.f1_score(valid_y_true, valid_y_pred, average='macro')}
                )
                valid_summary_writer.add_summary(valid_epoch_summary, global_step=i)
                logging.info('Macro-F1-score on va_data at epoch {0}: {1:.8f}'.format(i, metrics.f1_score(valid_y_true, valid_y_pred, average='macro')))
                # visualize and save labels and predictions of validation dataset
                if i % FLAGS.skip_epoch == 0:
                    pio.export_image(output_dir, i, valid_y_true, valid_y_pred)
                if i % (5 * FLAGS.skip_epoch) == 0:
                    fpo.export_valid_prediction(output_dir, i, valid_y_true, valid_y_pred)

            # pred on te_data
            sess.run(te_init)
            test_y_pred = []
            try:
                while True:
                    y_pred_ = sess.run(y_pred, feed_dict={self.keep_prob: 1.0})
                    y_pred_ = y_pred_.tolist()
                    test_y_pred.extend(y_pred_)
            except tf.errors.OutOfRangeError:
                pass
            fpo.export_test_prediction(output_dir, test_y_pred)
            logging.info('Prediction on test dataset is done...')


class deepLSTM(LSTM):

    def __init__(self, embedding_size, num_lstm_units, max_sen_len, batch_size, learning_rate, num_epoch, aspect_id, model_type, num_deep_rnn):
        super().__init__(embedding_size, num_lstm_units, max_sen_len, batch_size, learning_rate, num_epoch, aspect_id, model_type)
        self.num_deep_rnn = num_deep_rnn
        self.params['num_deep_rnn'] = self.num_deep_rnn  # new additional parameter

    def get_outputs(self, inputs, batch_sen_len):
        logging.info('dynamic_run in deeplstm is calling...')
        _, states = tdo.dynamic_deep_lstm(inputs, batch_sen_len, self.num_lstm_units, self.num_deep_rnn)
        last_eff_output = states[-1].h  # [B, H], hidden states in last layer
        return tf.nn.dropout(last_eff_output, keep_prob=self.keep_prob)


class biLSTM(LSTM):

    def get_outputs(self, inputs, batch_sen_len):
        logging.info('dynamic_run in bilstm is calling...')
        _, states = tdo.dynamic_bi_lstm(inputs, batch_sen_len, self.num_lstm_units)
        last_eff_output = tf.concat([states[0].h, states[1].h], axis=-1)  # [B, 2H]
        return tf.nn.dropout(last_eff_output, keep_prob=self.keep_prob)


class aeatLSTM(LSTM):

    def __init__(self, embedding_size, num_lstm_units, max_sen_len, batch_size, learning_rate, num_epoch, aspect_id, model_type, aspect_embedding_size):
        super().__init__(embedding_size, num_lstm_units, max_sen_len, batch_size, learning_rate, num_epoch, aspect_id, model_type)
        self.aspect_embedding_size = aspect_embedding_size
        self.params['aspect_embedding_size'] = self.aspect_embedding_size

    def get_inputs(self, batch_sen_encode, batch_sen_len):
        inputs_word = tf.nn.embedding_lookup(self.word_embedding, batch_sen_encode)  # [B, M, E]
        self.atten_vectors = tuo.add_atten_vectors(self.num_lstm_units, self.aspect_embedding_size)
        aspect_embedding_input = self.atten_vectors['aspect_embedding'][0]
        inputs_aspect = tuo.aspect_embedding_tile(aspect_embedding_input, batch_sen_len, self.max_sen_len)  # [B, M, AE]
        inputs = tf.concat([inputs_word, inputs_aspect], axis=-1)  # [B, M, E+AE]
        return tf.nn.dropout(inputs, keep_prob=self.keep_prob)
    
    def hard_attention(self, outputs, outputs_word, last_eff_output, topk=3):
        outputs_act = tf.nn.relu(outputs)  # relu activation max(0, x), [B, M, H+AE]
        outputs_act = tf.reshape(outputs_act, shape=[-1, self.num_lstm_units + self.aspect_embedding_size])  # [B*M, H+AE]
        weightsAtten = self.atten_vectors['attention']  # [H+AE]
        atten_weights = tf.expand_dims(weightsAtten, axis=-1)  # [H+AE, 1]
        atten_inputs = tf.reshape(tf.matmul(outputs_act, atten_weights), shape=[-1, self.max_sen_len])  # [B*M, 1] -> [B, M]
        atten_top = tf.nn.top_k(atten_inputs, k=topk)  # pick most important 3 words, find topk in each row
        atten_top_min = tf.reduce_min(atten_top.values, axis=1, keepdims=True)  # find min_val in topk
        atten_top_mask = tf.greater_equal(atten_inputs, atten_top_min)  # True for topk location only
        atten_top_mask = tf.cast(atten_top_mask, tf.float32)  # [B, M]
        atten_inputs_hard = tf.multiply(atten_inputs, atten_top_mask)
        atten_outputs = tf.nn.softmax(atten_inputs_hard, axis=1)

        atten_outputs = tf.expand_dims(atten_outputs, axis=-1)  # [B, M, 1]
        outputs_hidden = tf.transpose(outputs_word, perm=[0, 2, 1])  # [B, H, M]

        outputs_r = tf.reshape(tf.matmul(outputs_hidden, atten_outputs), shape=[-1, self.num_lstm_units])  # [B, H]
        weightsP = self.atten_vectors['projection'][2]  # [H, H]
        weightsX = self.atten_vectors['projection'][3]  # [H, H]
        outputs = tf.matmul(outputs_r, weightsP) + tf.matmul(last_eff_output, weightsX)  # [B, H]
        return tf.nn.tanh(outputs)

    def get_outputs(self, inputs, batch_sen_len):
        outputs_word, state = tdo.dynamic_basic_lstm(inputs, batch_sen_len, self.num_lstm_units)  # [B, M, H]
        last_eff_output = state.h  # [B, H]
        aspect_embedding_output = self.atten_vectors['aspect_embedding'][1]
        outputs_aspect = tuo.aspect_embedding_tile(aspect_embedding_output, batch_sen_len, self.max_sen_len)  # [B, M, AE]
        tmp_outputs = tuo.concat_word_aspect(outputs_word, outputs_aspect, self.atten_vectors, self.num_lstm_units, self.aspect_embedding_size, self.max_sen_len)
        tmp_outputs = tf.nn.dropout(tmp_outputs, keep_prob=self.keep_prob)  # [B, M, H+AE]
        outputs = self.hard_attention(tmp_outputs, outputs_word, last_eff_output)
        return tf.nn.dropout(outputs, keep_prob=self.keep_prob)


def main(_):
    lstmParams = {
        'embedding_size': FLAGS.embedding_size,
        'num_lstm_units': FLAGS.num_lstm_units,
        'max_sen_len': FLAGS.max_sen_len,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'num_epoch': FLAGS.num_epoch,
        'aspect_id': FLAGS.aspect_id,
        'model_type': FLAGS.model_type
    }

    if FLAGS.model_type == 'lstm':
        lstm = LSTM(**lstmParams)
    elif FLAGS.model_type == 'bilstm':
        lstm = biLSTM(**lstmParams)
    elif FLAGS.model_type == 'deeplstm':
        lstmParams['num_deep_rnn'] = FLAGS.num_deep_rnn
        lstm = deepLSTM(**lstmParams)
    elif FLAGS.model_type == 'aeatlstm':
        lstmParams['aspect_embedding_size'] = FLAGS.aspect_embedding_size
        lstm = aeatLSTM(**lstmParams)

    logging.info('\nRunning {} with apsect {}'.format(FLAGS.model_type, FLAGS.aspect_id))
    lstm.run()


if __name__ == '__main__':
    tf.app.run()