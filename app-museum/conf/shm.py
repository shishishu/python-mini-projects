#!/usr/bin/env python
#coding=utf-8
# @file  : shm
# @time  : 2/13/2022 6:24 PM
# @author: shishishu

base_info = {
    'museum': 'SHM'
}

levels_map = {
    '金石': ['青铜', '雕塑', '钱币', '其他'],
    '陶瓷': ['陶瓷'],
    '书画': ['书法', '绘画', '其他'],
    '工艺': ['玉器', '家具', '少数民族', '其他']
}

# 分页机制（防止一个目录下文件数过多）
raw_paging_org_conf = {
    1: {'main_key': 'num_id', 'use_hash': False, 'num_parts': 20},
    2: {'main_key': 'query_id', 'use_hash': True, 'num_parts': 10}
}