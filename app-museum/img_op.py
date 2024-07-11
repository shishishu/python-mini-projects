#!/usr/bin/env python
#coding=utf-8
# @file  : img_op
# @time  : 3/20/2022 5:41 PM
# @author: shishishu

import sys
import os
import json
import shutil
import numpy as np
from PIL import Image
from multiprocessing import Pool
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S')

IMG_T_SHAPE = (640, 640)

def rotate_concate_img(im, rotate_degrees, concate_axis):
    im_concate_data = None
    for degree in rotate_degrees:
        im_rotate = im.rotate(degree)
        if im_concate_data is None:
            im_concate_data = np.array(im_rotate)
        else:
            im_concate_data = np.concatenate((im_concate_data, np.array(im_rotate)), axis=concate_axis)
    return im_concate_data

def rotate_save_img(im, rotate_path, rotate_degrees, concate_axis=1):
    # concate_axis: =0, y方向; =1, x方向
    im_concate_data = rotate_concate_img(im, rotate_degrees, concate_axis)
    if os.path.exists(rotate_path):
        os.remove(rotate_path)
    Image.fromarray(im_concate_data).save(rotate_path)
    return

def load_rotate_img(img_path, img_reshape_size):
    img_dir, img_file = os.path.split(img_path)
    img_name = img_file.split('.')[0]
    im = Image.open(img_path)
    im = im.resize(img_reshape_size)
    # 顺时针
    rotate_degrees = [0, -90, -180, -270]
    rotate_path = os.path.join(img_dir, img_name + '_cw.jpg')
    rotate_save_img(im, rotate_path, rotate_degrees)

    # 逆时针
    rotate_degrees = [0, 90, 180, 270]
    rotate_path = os.path.join(img_dir, img_name + '_ccw.jpg')
    rotate_save_img(im, rotate_path, rotate_degrees)
    # logging.info("img is done: {}".format(img_path))
    return

def load_rotate_img_batch(img_paths, img_reshape_size=IMG_T_SHAPE):
    total_cnt = 0
    err_cnt = 0
    for img_path in img_paths:
        total_cnt += 1
        try:
            load_rotate_img(img_path, img_reshape_size)
        except:
            err_cnt += 1
            logging.error("fail to rotate img: {}".format(img_path))
    logging.info("total_cnt={}, err_cnt={}".format(total_cnt, err_cnt))
    return

def load_rotate_img_mp(img_paths, num_workers, img_reshape_size=IMG_T_SHAPE):
    pool = Pool(num_workers)
    pool.starmap(load_rotate_img, list(zip(img_paths, [img_reshape_size for i in range(len(img_paths))])))
    pool.close()
    pool.join()
    return

def load_rotate_spec_level_img(level_json_path, num_workes=1):
    level_data = json.load(open(level_json_path))
    img_paths = []
    for item_dict in level_data:
        if item_dict['has_img_thumbnail_path'] == 1:
            img_paths.append(item_dict['img_thumbnail_path'])
    if num_workes > 1:
        load_rotate_img_mp(img_paths, num_workes)
    else:
        load_rotate_img_batch(img_paths)
    return

def combine_img(img_paths, h_num, v_num):
    assert len(img_paths) == h_num * v_num, 'wrong num of img inputs'
    img_groups = defaultdict(list)
    for idx, img_path in enumerate(img_paths):
        group_id = idx % v_num
        img_groups[group_id].append(img_path)
    im_v_list = []
    for group_id, img_paths in img_groups.items():
        im_data = np.array(Image.open(img_paths[0]))
        for img_path in img_paths[1:]:
            im_data = np.concatenate((im_data, np.array(Image.open(img_path))), axis=1)
        im_v_list.append(im_data)

    final_im_data = im_v_list[0]
    for im_data in im_v_list[1:]:
        final_im_data = np.concatenate((final_im_data, im_data), axis=0)
    logging.info("shape of final_im_data is: {}".format(final_im_data.shape))
    return final_im_data

def combine_spec_level_img(level_json_path, dest_dir, img_type, h_num=4, v_num=9):
    dest_img_dir = os.path.join(dest_dir, img_type + '_' + str(h_num) + '_' + str(v_num))
    os.makedirs(dest_img_dir, exist_ok=True)

    level_data = json.load(open(level_json_path))
    img_paths = []
    for item_dict in level_data:
        if item_dict['has_img_thumbnail_path'] == 1:
            img_dir, img_file = os.path.split(item_dict['img_thumbnail_path'])
            img_name = img_file.split('.')[0]
            img_path = os.path.join(img_dir, img_name + '_' + img_type + '.jpg')
            if os.path.exists(img_path):
                img_paths.append(img_path)
    logging.info("img_type={}, num_img_paths={}".format(img_type, len(img_paths)))

    num_combine_imgs = h_num * v_num
    img_combine_list = []
    img_combine_idx = 0
    for img_path in img_paths:
        if len(img_combine_list) < num_combine_imgs:
            img_combine_list.append(img_path)
            continue
        im_combine_data = combine_img(img_combine_list, h_num, v_num)
        combine_img_path = os.path.join(dest_img_dir, str(img_combine_idx) + '.jpg')
        combine_json_path = os.path.join(dest_img_dir, str(img_combine_idx) + '.json')
        if os.path.exists(combine_img_path):
            os.remove(combine_img_path)
        Image.fromarray(im_combine_data).save(combine_img_path)
        json.dump(img_combine_list, open(combine_json_path, 'w'), ensure_ascii=False, indent=4)
        img_combine_list.clear()
        img_combine_idx += 1
        logging.info("img_combine_idx={}".format(img_combine_idx))
    return

def cp_target_images(level_json_path, dest_dir, key_words='青花'):
    os.makedirs(dest_dir, exist_ok=True)

    target_cnt = 0

    data = json.load(open(level_json_path))
    for img in data:
        if img['has_img_thumbnail_path'] == 0 or img['has_index_path'] == 0:
            continue
        img_thumbnail_path = img['img_thumbnail_path']
        index_path = img['index_path']
        content = open(index_path, encoding='utf-8').read()
        if key_words in content:
            img_name = os.path.split(img_thumbnail_path)[-1]
            img_path = os.path.join(dest_dir, img_name)
            shutil.copyfile(img_thumbnail_path, img_path)
            target_cnt += 1
    logging.info("totol cnt is: {}".format(target_cnt))
    return


if __name__ == '__main__':

    # img_path = '/mnt/e/app-museum/data/SHM/0/0/img/CI00004860_t.jpg'
    # load_rotate_img(img_path)

    # level_json_path = '/mnt/d/app-museum/data/SHM/TAOCI.json'
    # load_rotate_spec_level_img(level_json_path, 10)

    # level_json_path = '/mnt/d/app-museum/data/SHM/TAOCI.json'
    # dest_dir = '/mnt/d/app-museum/data/SHM/img/TAOCI'
    # combine_spec_level_img(level_json_path, dest_dir, 'cw', 4, 9)
    # combine_spec_level_img(level_json_path, dest_dir, 'ccw', 4, 9)
    #
    # combine_spec_level_img(level_json_path, dest_dir, 'cw', 2, 4)
    # combine_spec_level_img(level_json_path, dest_dir, 'ccw', 2, 4)

    level_json_path = '/mnt/d/app-museum/data/SHM/TAOCI.json'
    dest_dir = '/mnt/e/app-museum/data/SHM/TAOCI/TAOCI_t_qh'
    cp_target_images(level_json_path, dest_dir)
