#!/usr/bin/env python
#coding=utf-8
# @file  : dnnBase
# @time  : 5/31/2020 5:34 PM
# @author: shishishu

import os
import tensorflow as tf
import pandas as pd
from conf import config
from lib.utils.fileOperation import FileOperation
from lib.utils.generalNet import GeneralNet
from sklearn.metrics import accuracy_score
from lib.utils.utils import collect_log_content, collect_log_key_content, calculate_dist, calculate_delta_dist
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'dnn.log'),
                    filemode='a')


class DNNBase:

    def __init__(self, batch_size, learning_rate, l2_reg, num_epoch, skip_epoch, skip_step, fea_size, pred_domain, input_dir, model_type, model_version):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.num_epoch = num_epoch
        self.skip_epoch = skip_epoch
        self.skip_step = skip_step
        self.fea_size = fea_size
        self.pred_domain = pred_domain
        self.stan_dist_dict = config.AGE_DIST_DICT if self.pred_domain == 'age' else config.GENDER_DIST_DICT

        self.model_type = model_type
        model_name = '_'.join([self.model_type, self.pred_domain, model_version])
        self.model_dir = os.path.join(config.MODEL_DIR, model_name)
        FileOperation.safe_mkdir(self.model_dir)
        self.log_dir = os.path.join(config.LOG_DIR, model_name)
        FileOperation.safe_mkdir(self.log_dir)

        self.num_class = len(config.LABEL_TYPE_DICT[self.pred_domain])
        self.onehot_dim = config.ONEHOT_DIM_DICT.get(self.model_type, 0)
        self.input_dir = input_dir
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')

        self.log_dict = dict()
        self.log_path = os.path.join(self.model_dir, 'log_dict.json')

        # init
        self.feas, self.sen_encode, self.sen_encode2, self.sen_encode3, self.onehot_encdoe, self.sen_len, self.labels = [None, None, None, None, None, None, None]

    def get_data(self, *args, **kwargs):
        with tf.name_scope('data'):
            tr_file = os.path.join(self.input_dir, 'tr_' + self.pred_domain + '.txt')
            va_1_file = os.path.join(self.input_dir, 'va_1_' + self.pred_domain + '.txt')
            va_2_file = os.path.join(self.input_dir, 'va_2_' + self.pred_domain + '.txt')
            self.te_file = os.path.join(self.input_dir, 'te_' + self.pred_domain + '.txt')
            logging.info('tr file is: {}'.format(tr_file))
            logging.info('va_1 file is: {}'.format(va_1_file))
            logging.info('va_2 file is: {}'.format(va_2_file))
            logging.info('te file is: {}'.format(self.te_file))
            logging.info('Start read data...')
            if self.model_type in ['mlp']:
                decode_func = GeneralNet.decode_txt_vec
            elif self.model_type in ['rnn2', 'rnn2cate', 'rnn2gender', 'rnn2cross', 'rnn2concat', 'txs2']:
                decode_func = GeneralNet.decode_txt_seq2
            elif self.model_type in ['rnn3']:
                decode_func = GeneralNet.decode_txt_seq3
            else:
                decode_func = GeneralNet.decode_txt_seq  # rnn, txs
            tr_data = tf.data.TextLineDataset([tr_file]).map(lambda x: decode_func(x, self.num_class)).prefetch(10 * self.batch_size)
            tr_data = tr_data.shuffle(buffer_size=10*self.batch_size).batch(self.batch_size)
            va_1_data = tf.data.TextLineDataset([va_1_file]).map(lambda x: decode_func(x, self.num_class))
            va_1_data = va_1_data.batch(self.batch_size)
            va_2_data = tf.data.TextLineDataset([va_2_file]).map(lambda x: decode_func(x, self.num_class))
            va_2_data = va_2_data.batch(self.batch_size)
            te_data = tf.data.TextLineDataset([self.te_file]).map(lambda x: decode_func(x, self.num_class))
            te_data = te_data.batch(self.batch_size)
            logging.info('Start iterator...')
            iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
            batch_feas, batch_labels = iterator.get_next()

            self.tr_init = iterator.make_initializer(tr_data, name='tr_init')
            self.va_1_init = iterator.make_initializer(va_1_data, name='va_1_init')
            self.va_2_init = iterator.make_initializer(va_2_data, name='va_2_init')
            self.te_init = iterator.make_initializer(te_data, name='te_init')

            if self.model_type in ['mlp']:
                self.feas = tf.reshape(batch_feas, shape=[-1, self.fea_size])  # [B, F]
            else:
                self.sen_len = tf.reshape(batch_feas['sen_len'], shape=[-1])  # [B]
                self.sen_encode = tf.reshape(batch_feas['sen_encode'], shape=[-1, config.MAX_SEN_LEN])  # [B, M]
                if self.model_type in ['rnn2', 'rnn2cate', 'rnn2gender', 'rnn2cross', 'rnn2concat', 'rnn3', 'txs2']:
                    self.sen_encode2 = tf.reshape(batch_feas['sen_encode2'], shape=[-1, config.MAX_SEN_LEN])  # [B, M]
                    if self.model_type in ['rnn2cate', 'rnn2gender', 'rnn2cross', 'rnn2concat']:
                        self.onehot_encode = tf.reshape(batch_feas['onehot_encode'], shape=[-1, self.onehot_dim])  # [B, OH]
                    if self.model_type in ['rnn3']:
                        self.sen_encode3 = tf.reshape(batch_feas['sen_encode3'], shape=[-1, config.MAX_SEN_LEN])  # [B, M]
            self.labels = tf.reshape(batch_labels, shape=[-1, self.num_class])  # [B, C]

    def inference(self, *args, **kwargs):
        pass

    def loss(self, *args, **kwargs):
        with tf.name_scope('loss'):
            engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)  # use logits
            self.loss = tf.reduce_mean(engin_loss)
            if self.l2_reg > 0:
                vars = tf.trainable_variables()
                # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * self.l2_reg
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_reg  # exclude bias
                self.loss += l2_loss
        return

    def optimize(self, *args, **kwargs):
        with tf.name_scope('opt'):
            self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.gstep)
        return

    def eval(self, *args, **kwargs):
        with tf.name_scope('predict'):

            self.y_true = tf.argmax(self.labels, axis=1)
            self.prob = tf.nn.softmax(self.logits, axis=1)
            self.y_pred = tf.argmax(self.prob, axis=1)
            correct_preds = tf.equal(self.y_true, self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return

    def summary(self, *args, **kwargs):
        with tf.name_scope('batch_based'):
            loss_summary = tf.summary.scalar('loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        with tf.name_scope('epoch_based'):  # summary op after one epoch
            self.epoch_loss = tf.placeholder(tf.float32, shape=None, name='epoch_loss')
            self.epoch_accuracy = tf.placeholder(tf.float32, shape=None, name='epoch_accuracy')
            epoch_loss_summary = tf.summary.scalar('epoch_loss', self.epoch_loss)
            epoch_accuracy_summary = tf.summary.scalar('epoch_accuracy', self.epoch_accuracy)
            self.epoch_summary_op = tf.summary.merge([epoch_loss_summary, epoch_accuracy_summary])
        return

    def build(self, *args, **kwargs):
        # build the computation graph
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step, *args, **kwargs):
        logging.info('epoch is {}'.format(epoch))
        sess.run(init)
        total_loss = 0
        n_batches = 0
        logging.info('start training...')
        train_y_true = []
        train_y_pred = []
        try:
            while True:
                _, _loss, _summary, _y_true, _y_pred = sess.run([self.train_op, self.loss, self.summary_op, self.y_true, self.y_pred], feed_dict={self.keep_prob: 0.8})
                writer.add_summary(_summary, global_step=step)
                _y_true = _y_true.tolist()
                _y_pred = _y_pred.tolist()
                train_y_true.extend(_y_true)
                train_y_pred.extend(_y_pred)
                if (step + 1) % self.skip_step == 0:
                    logging.info('loss at step {}: {:.4f}'.format(step + 1, _loss))
                step += 1
                total_loss += _loss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        epoch_summary = sess.run(
            self.epoch_summary_op,
            feed_dict={self.epoch_loss: total_loss / n_batches, self.epoch_accuracy: accuracy_score(train_y_true, train_y_pred)}
        )
        writer.add_summary(epoch_summary, global_step=epoch)  # record as whole epoch
        if epoch % (10 * self.skip_epoch) == 0:
            saver.save(sess, os.path.join(self.model_dir, self.__class__.__name__), global_step=(step+1))  # prefix: ConvNet/ConvNet-...
        logging.info('average loss at epoch {}: {:.4f}'.format(epoch, total_loss/n_batches))
        return step

    def eval_epoch(self, sess, va_type, init, writer, epoch, step, *args, **kwargs):
        logging.info('current eval type is: {}'.format(va_type))
        keep_prob = 0.8 if va_type == 'va_1' else 1.0  # va_1: eval, va_2: test
        sess.run(init)
        total_loss = 0
        n_batches = 0
        eval_y_true = []
        eval_y_pred = []
        try:
            while True:
                _loss, _y_true, _y_pred, = sess.run([self.loss, self.y_true, self.y_pred], feed_dict={self.keep_prob: keep_prob})
                _y_true = _y_true.tolist()
                _y_pred = _y_pred.tolist()
                eval_y_true.extend(_y_true)
                eval_y_pred.extend(_y_pred)
                total_loss += _loss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        epoch_summary = sess.run(
            self.epoch_summary_op,
            feed_dict={self.epoch_loss: total_loss / n_batches, self.epoch_accuracy: accuracy_score(eval_y_true, eval_y_pred)}
        )
        writer.add_summary(epoch_summary, global_step=epoch)  # record as whole epoch
        if va_type == 'va_1':
            logging.info('average loss at epoch {} in eval: {:.4f}'.format(epoch, total_loss / n_batches))
        else:
            logging.info('eval accuracy at epoch {}: {:.4f}'.format(epoch, accuracy_score(eval_y_true, eval_y_pred)))
            collect_log_content(self.log_dict, 'eval accuracy at epoch {}: {:.4f}'.format(epoch, accuracy_score(eval_y_true, eval_y_pred)), self.log_path)
            eval_results = [i + 1 for i in eval_y_pred]  # add 1
            pred_col = 'predicted_' + self.pred_domain
            df_exp = pd.DataFrame(data={'user_id': list(range(len(eval_results))), pred_col: eval_results}, columns=['user_id', pred_col])
            pred_dist_dict = calculate_dist(df_exp, pred_col)
            logging.info('eval dist of {} is: {}'.format(self.pred_domain, pred_dist_dict))
            collect_log_key_content(self.log_dict, 'eval_dist_' + str(epoch), pred_dist_dict, self.log_path)
            delta_dist_dict = calculate_delta_dist(self.stan_dist_dict, pred_dist_dict)
            logging.info('eval delta dist of {} is: {}'.format(self.pred_domain, delta_dist_dict))
            collect_log_key_content(self.log_dict, 'eval_delta_dist_' + str(epoch), delta_dist_dict, self.log_path)
        return

    def predict_result(self, sess, init, *args, **kwargs):
        # pred on te_data
        sess.run(init)  # te.init
        test_y_pred = []
        try:
            while True:
                _y_pred = sess.run(self.y_pred, feed_dict={self.keep_prob: 1.0})
                _y_pred = _y_pred.tolist()
                test_y_pred.extend(_y_pred)
        except tf.errors.OutOfRangeError:
            pass
        df_test = FileOperation.load_csv(self.te_file, sep=' ', has_header=False)
        logging.info('shape of df_test is: {}'.format(df_test.shape))
        pred_results = [i + 1 for i in test_y_pred]  # add 1
        pred_col = 'predicted_' + self.pred_domain
        df_exp = pd.DataFrame(data={'user_id': list(df_test[0]), pred_col: pred_results}, columns=['user_id', pred_col])
        FileOperation.save_csv(df_exp, os.path.join(self.model_dir, self.pred_domain + '_export.csv'))
        pred_dist_dict = calculate_dist(df_exp, pred_col)
        logging.info('pred dist of {} is: {}'.format(self.pred_domain, pred_dist_dict))
        collect_log_content(self.log_dict, 'pred dist of {} is: {}'.format(self.pred_domain, pred_dist_dict), self.log_path)
        delta_dist_dict = calculate_delta_dist(self.stan_dist_dict, pred_dist_dict)
        logging.info('pred delta dist of {} is: {}'.format(self.pred_domain, delta_dist_dict))
        collect_log_content(self.log_dict, 'pred delta dist of {} is: {}'.format(self.pred_domain, delta_dist_dict), self.log_path)
        logging.info('Prediction on test dataset is done...')
        return

    def run(self, *args, **kwargs):
        # avoid error: CUDNN_STATUS_INTERNAL_ERROR
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=sess_config) as sess:
            train_log_dir = os.path.join(self.log_dir, 'train')
            valid_log_dir = os.path.join(self.log_dir, 'valid')
            valid2_log_dir = os.path.join(self.log_dir, 'valid2')
            FileOperation.safe_mkdir(train_log_dir)
            FileOperation.safe_mkdir(valid_log_dir)
            FileOperation.safe_mkdir(valid2_log_dir)
            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            valid_summary_writer = tf.summary.FileWriter(valid_log_dir)  # no sess.graph in valid
            valid2_summary_writer = tf.summary.FileWriter(valid2_log_dir)  # no sess.graph in valid
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # restore model
            ckpt = tf.train.get_checkpoint_state(self.model_dir, latest_filename='checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                logging.info('ckpt is: ', ckpt)
                saver.restore(sess, ckpt.model_checkpoint_path)
                logging.info('loading saved model: {}'.format(ckpt.model_checkpoint_path))
            else:
                logging.info('new model is created...')
            step = self.gstep.eval()
            for epoch in range(1, self.num_epoch + 1):
                step = self.train_one_epoch(sess, saver, self.tr_init, train_summary_writer, epoch, step)
                if epoch % self.skip_epoch == 0:
                    self.eval_epoch(sess, 'va_1', self.va_1_init, valid_summary_writer, epoch, step)
                    self.eval_epoch(sess, 'va_2', self.va_2_init, valid2_summary_writer, epoch, step)
            self.predict_result(sess, self.te_init)
            train_summary_writer.close()
            valid_summary_writer.close()
            valid2_summary_writer.close()
        return

    @staticmethod
    def default_params():
        params = dict()
        params['fea_size'] = 64
        params['pred_domain'] = 'age'
        params['batch_size'] = 4096
        params['learning_rate'] = 1e-4
        params['l2_reg'] = 0  # no regularization as default
        params['num_epoch'] = 50
        params['skip_epoch'] = 1
        params['skip_step'] = 10
        return params