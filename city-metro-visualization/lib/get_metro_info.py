#!/usr/bin/env python
#coding=utf-8
# @file  : get_metro_info
# @time  : 3/7/2020 2:23 PM
# @author: shishishu

import json
import requests
import os
import pandas as pd
from conf import config


class GetMetroInfo:

    def __init__(self, cityname_zh):
        self.data_dir = os.path.join(config.data_dir, cityname_zh)
        os.makedirs(self.data_dir, exist_ok=True)
        self.id = config.AMAP_CITY_DICT[cityname_zh]['id']
        self.cityname = config.AMAP_CITY_DICT[cityname_zh]['cityname']

    def get_page_data(self):
        # ref to: https://www.sohu.com/a/316351904_752099
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        url = config.AMAP_REQ_URL_PREFIX + self.id + '_drw_' + self.cityname + '.json'
        response = requests.get(url=url, headers=headers)
        html = response.text
        results = json.loads(html)

        data = []
        for line in results['l']:
            line_name = line['ln']
            line_name_add = line['la'] if line['la'] else 'default'
            line_color = '#' + line['cl']
            station_idx = 0
            for station in line['st']:
                station_idx += 1
                station_name = station['n']
                station_pinyin = station['sp']
                if line_name and station_name:
                    station_lng, station_lat = station['sl'].split(',')
                    # print(line_name, line_name_add, station_name, station_idx, station_lng, station_lat)
                    data.append([line_name, line_name_add, line_color, station_name, station_pinyin, station_idx, station_lng, station_lat])
        return data

    def dump_structured_data(self):
        data = self.get_page_data()
        file_path = os.path.join(self.data_dir, config.FILE_NAME_DICT['basic'])
        df = pd.DataFrame(
            data=data,
            columns=['line_name', 'line_name_add', 'line_color', 'station_name', 'station_pinyin', 'station_idx', 'station_lng', 'station_lat']
        )
        print('df shape is: ', df.shape)
        df.to_excel(file_path, index=False)
        return


if __name__ == '__main__':

    getMetInfer = GetMetroInfo('上海')
    getMetInfer.dump_structured_data()