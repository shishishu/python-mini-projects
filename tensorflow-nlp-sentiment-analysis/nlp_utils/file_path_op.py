#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 20 Apr 2019
@author: Nano Zhou
"""

import os
import glob


def get_dir_file_name(file_path):
    dir_name, file_name = os.path.split(file_path)
    file_name_stem, file_name_suffix = file_name.split('.')
    return dir_name, file_name_stem, file_name_suffix

def get_unprocessed_files(target_dir, file_type='txt'):
    file_path_list = []
    glob_files = glob.glob(target_dir + '/*.' + file_type)
    for file_path in glob_files:
        _, file_name_stem, _ = get_dir_file_name(file_path)
        tmp = file_name_stem.split('_')
        if tmp[-1] != str(1):  # ending by '_1' means processed already
            file_path_list.append(file_path)
    return file_path_list

def rename_processed_file(file_path):
    dir_name, file_name_stem, file_name_suffix = get_dir_file_name(file_path)
    file_name_stem = file_name_stem + '_1'  # use '_1' to mask as processed
    new_file_name = file_name_stem + '.' + file_name_suffix
    new_file_path = os.path.join(dir_name, new_file_name)
    os.rename(file_path, new_file_path)

def gene_dirs(model_dir, log_dir, output_dir, model_type, aspect_id, timestamp):
    spec_model_dir = os.path.join(model_dir, model_type, 'aspect_' + str(aspect_id), timestamp)
    spec_log_dir = os.path.join(log_dir, model_type, 'aspect_' + str(aspect_id), timestamp)
    spec_output_dir = os.path.join(output_dir, model_type, 'aspect_' + str(aspect_id), timestamp)
    if not os.path.exists(spec_model_dir):
        os.makedirs(spec_model_dir)
    if not os.path.exists(spec_log_dir):
        os.makedirs(spec_log_dir)
    if not os.path.exists(spec_output_dir):
        os.makedirs(spec_output_dir)
    sepc_runlog_dir = os.path.join(log_dir, model_type, 'aspect_' + str(aspect_id))
    train_log_dir = os.path.join(spec_log_dir, 'train')
    valid_log_dir = os.path.join(spec_log_dir, 'valid')
    if not os.path.exists(train_log_dir):
        os.mkdir(train_log_dir)
        os.mkdir(valid_log_dir)
    dirs = {
        'model_dir': spec_model_dir, 
        'output_dir': spec_output_dir,
        'runlog_dir': sepc_runlog_dir,
        'train_log_dir': train_log_dir,
        'valid_log_dir': valid_log_dir
    }
    return dirs

def save_retrival_file(file_path, timestamp, params):
    with open(file_path, 'a') as fa:
        fa.write('folder_name:{}\t'.format(timestamp))
        for key, val in params.items():
            fa.write(key + ': {}\t'.format(val))
        fa.write('\n')
    return

def export_valid_prediction(output_dir, epoch, y_true, y_pred):
    with open(output_dir + '/pred_epoch_' + str(epoch) + '.txt', 'w') as fw:
        fw.writelines(['{}\t{}\n'.format(a, b) for a, b in zip(y_true, y_pred)])
    return

def export_test_prediction(output_dir, y_pred):
    with open(output_dir + '/pred_test.txt', 'w') as fw:
        fw.writelines(['{}\n'.format(a) for a in y_pred])
    return