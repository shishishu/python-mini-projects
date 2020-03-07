#!/usr/bin/env python
#coding=utf-8
# @file  : config
# @time  : 3/7/2020 2:17 PM
# @author: shishishu

import os

current_path = os.path.abspath(__file__)
master_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + '..')
data_dir = master_path + './data'

# http://map.amap.com/subway/index.html
AMAP_CITY_DICT = {
    '北京': {'id': '1100', 'cityname': 'beijing', 'lng': 116.4074, 'lat': 39.9042},
    '上海': {'id': '3100', 'cityname': 'shanghai', 'lng': 121.4737, 'lat': 31.2304},
    '广州': {'id': '4401', 'cityname': 'guangzhou', 'lng': 113.2644, 'lat': 23.1291},
    '深圳': {'id': '4403', 'cityname': 'shenzhen', 'lng': 114.0579, 'lat': 22.5431}
}

AMAP_REQ_URL_PREFIX = 'http://map.amap.com/service/subway?_1555502190153&srhdata='

FILE_NAME_DICT = {
    'basic': 'basic_metro_info.xlsx',
    'exchange': 'metro_exchange_lines.xlsx',
    'adj': 'metro_adj_stations.xlsx'
}