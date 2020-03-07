#!/usr/bin/env python
#coding=utf-8
# @file  : get_data
# @time  : 3/7/2020 6:30 PM
# @author: shishishu

import argparse
from lib.get_metro_info import GetMetroInfo
from lib.post_metro_stats import MetroStats

parser = argparse.ArgumentParser()
parser.add_argument('--cityname_zh', type=str, default='上海', required=True, help='chinese name of targeted city')

def proc_data(cityname_zh):
    # dump basic data
    getMetInfer = GetMetroInfo(cityname_zh)
    getMetInfer.dump_structured_data()

    # post statistics
    metStaer = MetroStats(cityname_zh)
    metStaer.find_exchange_lines()
    metStaer.find_adj_stations([1, 3])  # adj 1km, 3km
    metStaer.plot_stats()
    return


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()
    proc_data(FLAGS.cityname_zh)
    print('proc data task is done...')