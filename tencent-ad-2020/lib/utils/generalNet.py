#!/usr/bin/env python
#coding=utf-8
# @file  : nets
# @time  : 5/31/2020 4:38 PM
# @author: shishishu

import tensorflow as tf
import pickle
import os
from conf import config

class GeneralNet:

    @staticmethod
    def decode_txt_vec(line, num_class): # decode line by line
        columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
        # user_id, feas, labels
        feas = tf.string_to_number(columns.values[1:(-1*num_class)], out_type=tf.float32)
        labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
        return feas, labels

    @staticmethod
    def decode_txt_seq(line, num_class): # decode line by line
        columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
        # user_id, feas-dict, labels
        sen_encode = tf.string_to_number(columns.values[1:int(-1*num_class - 1)], out_type=tf.int32)
        sen_len = tf.string_to_number(columns.values[int(-1*num_class - 1)], out_type=tf.int32)
        labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
        feas = {'sen_len': sen_len, 'sen_encode': sen_encode}
        return feas, labels

    @staticmethod
    def decode_txt_seq2(line, num_class): # decode line by line
        max_sen_len = config.MAX_SEN_LEN
        columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
        # user_id, feas-dict, labels
        sen_encode = tf.string_to_number(columns.values[1:int(1 + max_sen_len)], out_type=tf.int32)
        sen_len = tf.string_to_number(columns.values[int(1 + max_sen_len)], out_type=tf.int32)
        sen_encode2 = tf.string_to_number(columns.values[int(2 + max_sen_len): int(2 + 2*max_sen_len)], out_type=tf.int32)
        # labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
        labels = tf.string_to_number(columns.values[int(3 + 2*max_sen_len): int(3 + 2*max_sen_len + num_class)], out_type=tf.float32)
        onehot_encode = tf.string_to_number(columns.values[int(3 + 2*max_sen_len + num_class):], out_type=tf.float32)
        feas = {'sen_len': sen_len, 'sen_encode': sen_encode, 'sen_encode2': sen_encode2, 'onehot_encode': onehot_encode}
        return feas, labels

    @staticmethod
    def decode_txt_seq3(line, num_class): # decode line by line
        max_sen_len = config.MAX_SEN_LEN
        columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
        # user_id, feas-dict, labels
        sen_encode = tf.string_to_number(columns.values[1:int(1 + max_sen_len)], out_type=tf.int32)
        sen_len = tf.string_to_number(columns.values[int(1 + max_sen_len)], out_type=tf.int32)
        sen_encode2 = tf.string_to_number(columns.values[int(2 + max_sen_len): int(2 + 2*max_sen_len)], out_type=tf.int32)
        sen_encode3 = tf.string_to_number(columns.values[int(3 + 2*max_sen_len): int(3 + 3*max_sen_len)], out_type=tf.int32)
        labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
        feas = {'sen_len': sen_len, 'sen_encode': sen_encode, 'sen_encode2': sen_encode2, 'sen_encode3': sen_encode3}
        return feas, labels

    @staticmethod
    def fully_connected_layer(inputs, out_dim, activation='relu', keep_prob=1.0, scope_name='fc'):
        # inputs: [B, I]
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_dim = inputs.shape[-1]
            W = tf.get_variable(
                name='weights',
                shape=[in_dim, out_dim],  # [I, O]
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b = tf.get_variable(
                name='biases',
                shape=[out_dim],
                initializer=tf.random_uniform_initializer(-0.01, 0.01)
            )
            output = tf.matmul(inputs, W) + b  # [B, H]
            acti_func = GeneralNet.get_activation(activation)
            if acti_func:
                output = acti_func(output)
        return tf.nn.dropout(output, keep_prob, name=scope.name)

    @staticmethod
    def fully_connected_layers(inputs, out_units, activation='relu', keep_prob=1.0, scope_name='fc'):
        tmp_inputs = inputs
        for idx, out_dim in enumerate(out_units):
            tmp_inputs = GeneralNet.fully_connected_layer(tmp_inputs, out_dim, activation, keep_prob, (scope_name + '_' + str(idx)))
        return tmp_inputs

    @staticmethod
    def get_activation(activation_string):
        if not activation_string:
            return None
        act = activation_string.lower()
        if act == "identity":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "tanh":
            return tf.nn.tanh
        elif act == 'sigmoid':
            return tf.nn.sigmoid
        else:
            raise ValueError("Unsupported activation: %s" % act)

    @staticmethod
    def softmax_classifier(inputs, num_class, scope_name='softmax'):
        # inputs: [B, H]
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_dim = inputs.shape[-1]
            W = tf.get_variable(
                name='softmax_W',
                shape=[in_dim, num_class],  # [B, C]
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b = tf.get_variable(
                name='softmax_b',
                shape=[num_class],
                initializer=tf.random_uniform_initializer(-0.01, 0.01)
            )
            pred = tf.matmul(inputs, W) + b  # [B, C]
        return tf.nn.softmax(pred, axis=1, name=scope.name)

    @staticmethod
    def get_embedding_vectors(embedding_size, ad_domain='creative_id'):
        with tf.variable_scope('embeddings_' + ad_domain):
            # Dummy-0
            dummy_embedding = tf.get_variable(
                name='dummy',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=False
            )
            # CLS-1
            cls_embedding = tf.get_variable(
                name='cls',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True
            )
            # UNK-2
            unk_embedding = tf.get_variable(
                name='unk',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True
            )
            # use pretrained embedding for frequent words
            model_dir = os.path.join(config.MODEL_DIR, 'w2v_' + str(embedding_size) + '_' + ad_domain)
            embedding_file_path = os.path.join(model_dir, 'w2v_embed_' + str(embedding_size) + '.pkl')
            pretrained_embedding = pickle.load(open(embedding_file_path, 'rb'))
            pretrained_embedding = tf.get_variable(
                name='pretrained',
                shape=pretrained_embedding.shape,
                initializer=tf.constant_initializer(pretrained_embedding, dtype=tf.float32),
                trainable=False
            )
        word_embedding = tf.concat([dummy_embedding, cls_embedding, unk_embedding, pretrained_embedding], axis=0)
        return word_embedding

    @staticmethod
    def get_embedding_vectors_trainable(embedding_size):
        with tf.variable_scope('embeddings'):
            # Dummy-0
            dummy_embedding = tf.get_variable(
                name='dummy',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=False
            )
            # CLS-1
            cls_embedding = tf.get_variable(
                name='cls',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True
            )
            # UNK-2
            unk_embedding = tf.get_variable(
                name='unk',
                shape=[1, embedding_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True
            )
            # use pretrained embedding for frequent words
            model_dir = os.path.join(config.MODEL_DIR, 'w2v_' + str(embedding_size))
            embedding_file_path = os.path.join(model_dir, 'w2v_embed_' + str(embedding_size) + '.pkl')
            pretrained_embedding = pickle.load(open(embedding_file_path, 'rb'))
            pretrained_embedding = tf.get_variable(
                name='pretrained',
                shape=pretrained_embedding.shape,
                initializer=tf.constant_initializer(pretrained_embedding, dtype=tf.float32),
                trainable=True
            )
        word_embedding = tf.concat([dummy_embedding, cls_embedding, unk_embedding, pretrained_embedding], axis=0)
        return word_embedding

    @staticmethod
    def gene_big_group_labels(labels, dict_data, num_group):
        y_true = tf.argmax(labels, axis=1)  # [B]
        y_new = tf.gather(dict_data, y_true)  # [B]
        y_out = tf.one_hot(y_new, depth=num_group)
        return y_out

    @staticmethod
    def focal_loss_softmax(labels, logits, gamma=2):
        """
        Computer focal loss for multi classification
        Args:
          labels: A flaot32 tensor of shape [batch_size, num_classes].
          logits: A float32 tensor of shape [batch_size,num_classes].
          gamma: A scalar for focal loss gamma hyper-parameter.
        Returns:
          A tensor of the same shape as `lables`
        """
        y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
        # labels = tf.one_hot(labels, depth=y_pred.shape[1])
        L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
        # L = tf.reduce_sum(L, axis=1)
        return L