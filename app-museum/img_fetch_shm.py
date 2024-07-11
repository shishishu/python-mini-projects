#!/usr/bin/env python
#coding=utf-8
# @file  : img_fetch_shm
# @time  : 2/13/2022 8:30 PM
# @author: shishishu

import os
import sys
import eventlet
eventlet.monkey_patch()
import json
import time
import random
from lib.ImgFetchSHM import ImgFetchSHM
from lib.Utils import Utils
from conf.config import data_dir, archive_dir_linux as archive_dir

run_in_linux = True

BASE_INFO = {
    'museum': 'SHM',
    'level_name': Utils.char2pinyin('书画'),
    'sub_level_name': Utils.char2pinyin('绘画')
}

QUERY_INFO = {
    'web_url': 'https://www.shanghaimuseum.net/mu/frontend/pg/article/id/',
    'img_url_prefix': 'https://www.shanghaimuseum.net/mu/',
    'base_dir': os.path.join(archive_dir, BASE_INFO['museum'])
}

def time_out(interval, callback):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                with eventlet.Timeout(interval, True):
                    result = func(*args, **kwargs)
                    return result
            except eventlet.Timeout as e:
                callback(e)
        return wrapper
    return decorator

def timeout_callback(e):
    raise TimeoutError("time limit is: {}".format(e))


def build_query(query_id, worker_index):
    query = dict()
    query['id'] = query_id
    query['web_url'] = QUERY_INFO['web_url'] + query_id
    query['img_url_prefix'] = QUERY_INFO['img_url_prefix']
    # query['base_dir'] = os.path.join(data_dir, BASE_INFO['museum'], BASE_INFO['level_name'], BASE_INFO['sub_level_name'])
    sub_index = hash(query_id) % 10
    query['base_dir'] = os.path.join(QUERY_INFO['base_dir'], str(worker_index), str(sub_index))
    return query

def build_query_batch(query_ids, worker_index):
    query_list = []
    for query_id in query_ids:
        query_list.append(build_query(query_id, worker_index))
    return query_list

@time_out(10, timeout_callback)
def fetch_resp(query):
    imgFetchSHM = ImgFetchSHM(query['id'], query['web_url'], query['img_url_prefix'], query['base_dir'])
    imgFetchSHM.run()
    return

def fetch_resp_batch(query_list, worker_index):
    total_cnt = 0
    err_cnt = 0
    succ_id_list = []
    err_id_list = []
    err_id_list_history = json.load(open(os.path.join(QUERY_INFO['base_dir'], 'errid_3rd_worker_' + str(worker_index) + '.json'), 'r'))
    print("history: worker_index={}, num_err_id_list={}".format(worker_index, len(err_id_list_history)), flush=True)
    for query in query_list:
        if query['id'] not in err_id_list_history:
            continue
        total_cnt += 1
        if total_cnt % 10 == 0:
            print("current progress: total_cnt={}, err_cnt={}".format(total_cnt, err_cnt), flush=True)
        try:
            fetch_resp(query)
            succ_id_list.append(query['id'])
        except:
            err_cnt += 1
            err_id_list.append(query['id'])
        # sleep randomly
        time.sleep(random.random())
    print('finally: total_cnt={}, err_cnt={}'.format(total_cnt, err_cnt), flush=True)
    json.dump(succ_id_list, open(os.path.join(QUERY_INFO['base_dir'], 'succid_4th_worker_' + str(worker_index) + '.json'), 'w'))
    json.dump(err_id_list, open(os.path.join(QUERY_INFO['base_dir'], 'errid_4th_worker_' + str(worker_index) + '.json'), 'w'))
    return


if __name__ == '__main__':

    # query_ids = ['CI00010003', 'CI00009999', 'CI00005591', 'CI00000823']
    query_ids = ['CI000' + str(x).zfill(5) for x in range(0, 20000)]
    worker_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    worker_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if worker_num > 1:
        query_ids = [x for idx, x in enumerate(query_ids) if idx % worker_num == worker_index]
    query_list = build_query_batch(query_ids, worker_index)
    fetch_resp_batch(query_list, worker_index)