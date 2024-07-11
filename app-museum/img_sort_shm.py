#!/usr/bin/env python
# coding=utf-8
# @file  : img_fetch_shm
# @time  : 2/13/2022 8:30 PM
# @author: shishishu

import os
import sys
import json
from collections import defaultdict
from lib.Utils import Utils
from conf.config import data_dir_linux as data_dir, archive_dir_linux as archive_dir
from conf.shm import raw_paging_org_conf, levels_map, base_info
from bs4 import BeautifulSoup

import logging
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S')

run_in_linux = True

raw_data_dir = os.path.join(archive_dir, base_info['museum'])
meta_data_dir = os.path.join(data_dir, base_info['museum'])
os.makedirs(meta_data_dir, exist_ok=True)


def fetch_file_paths(query_id, dir_path):
    html_path = os.path.join(dir_path, 'html', query_id + '.html')
    img_origin_path = os.path.join(dir_path, 'img', query_id + '_o.jpg')
    img_thumbnail_path = os.path.join(dir_path, 'img', query_id + '_t.jpg')
    index_path = os.path.join(dir_path, 'index', query_id + '.index')
    index_abbr_path = os.path.join(dir_path, 'index', query_id + '.index.abbr')

    item_dict = dict()
    if os.path.exists(html_path):
        item_dict['query_id'] = query_id
        item_dict['html_path'] = html_path
    else:
        return item_dict
    if os.path.exists(img_origin_path):
        item_dict['has_img_origin_path'] = 1
        item_dict['img_origin_path'] = img_origin_path
    else:
        item_dict['has_img_origin_path'] = 0
    if os.path.exists(img_thumbnail_path):
        item_dict['has_img_thumbnail_path'] = 1
        item_dict['img_thumbnail_path'] = img_thumbnail_path
    else:
        item_dict['has_img_thumbnail_path'] = 0
    if os.path.exists(index_path):
        item_dict['has_index_path'] = 1
        item_dict['index_path'] = index_path
    else:
        item_dict['has_index_path'] = 0
    if os.path.exists(index_abbr_path):
        item_dict['has_index_abbr_path'] = 1
        item_dict['index_abbr_path'] = index_abbr_path
    else:
        item_dict['has_index_abbr_path'] = 0
    return item_dict

def fill_item_categories(item_dict):
    soup = BeautifulSoup(open(item_dict['html_path'], 'rb'), 'html.parser', from_encoding='urf-8')
    level_info = soup.find('ul', class_='shmu-breadcrumb-nav').get_text().split()
    if len(level_info) < 4 or len(level_info) > 5:
        logging.error("wrong category parsing with level_info: {}".format(level_info))
        return
    level_name = level_info[3]
    sub_level_name = level_info[4] if len(level_info) == 5 else ''

    # double check parsing
    if level_name not in levels_map or sub_level_name not in levels_map[level_name]:
        logging.warning("wrong category matching with level_info: {}".format(level_info))
        return
    item_dict['level_name'] = Utils.char2pinyin(level_name)
    item_dict['sub_level_name'] = Utils.char2pinyin(sub_level_name) if sub_level_name else sub_level_name
    return

def fetch_html_paths(dir_path):
    html_paths_dict = dict()
    first_level_parts = raw_paging_org_conf[1]['num_parts']
    second_level_parts = raw_paging_org_conf[2]['num_parts']
    for i in range(0, first_level_parts):
        for j in range(0, second_level_parts):
            base_dir = os.path.join(dir_path, str(i), str(j))
            html_dir = os.path.join(base_dir, 'html')
            for html_file in os.listdir(html_dir):
                query_id = html_file.split('.')[0]
                html_paths_dict[query_id] = base_dir
    return html_paths_dict

def fetch_item_info(query_id, base_dir):
    item_dict = fetch_file_paths(query_id, base_dir)
    fill_item_categories(item_dict)
    return item_dict

def fetch_item_info_batch(html_paths_dict):
    out_dict = dict()
    for query_id, html_path in html_paths_dict.items():
        try:
            item_dict = fetch_item_info(query_id, html_path)
            out_dict[query_id] = item_dict
        except:
            logging.warning("fail to fetch item info with query_id={}".format(query_id))
    return out_dict

def gen_meta():
    html_paths_dict = fetch_html_paths(raw_data_dir)
    info_dict = fetch_item_info_batch(html_paths_dict)

    level_dict = defaultdict(list)
    meta_dict = dict()

    # meta_dict['levels_map'] = levels_map
    # meta_dict['raw_paging_org'] = raw_paging_org_conf

    for id, item_dict in info_dict.items():
        if 'level_name' in item_dict and 'sub_level_name' in item_dict:
            query_id = item_dict['query_id']
            level_name = item_dict['level_name']
            level_dict[level_name].append(item_dict)

    for level_name, item_list in level_dict.items():
        level_meta_path = os.path.join(meta_data_dir, level_name + '.json')
        json.dump(item_list, open(level_meta_path, 'w'), ensure_ascii=False, indent=4)
        logging.info("level_name={}, len_item_list={}".format(level_name, len(item_list)))

        item_stat = defaultdict(int)
        for item_dict in item_list:
            item_stat['num_item'] += 1
            item_stat['has_img_origin_path'] += item_dict['has_img_origin_path']
            item_stat['has_img_thumbnail_path'] += item_dict['has_img_thumbnail_path']
            item_stat['has_index_path'] += item_dict['has_index_path']
            item_stat['has_index_abbr_path'] += item_dict['has_index_abbr_path']
        meta_dict[level_name] = item_stat
    meta_path = os.path.join(meta_data_dir, 'meta.json')
    json.dump(meta_dict, open(meta_path, 'w'), ensure_ascii=False, indent=4)
    return


if __name__ == '__main__':

    gen_meta()
    # item_dict = fetch_item_info('CI00006500', '/mnt/e/app-museum/data/SHM/0/0')
    # print(item_dict)