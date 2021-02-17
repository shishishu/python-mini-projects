#!/usr/bin/env python
#coding=utf-8
# @file  : post_metro
# @time  : 3/7/2020 3:12 PM
# @author: shishishu

import os
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
from conf import config


class MetroStats:

    def __init__(self, cityname_zh):
        self.data_dir = os.path.join(config.data_dir, cityname_zh)

    def find_exchange_lines(self):
        df = pd.read_excel(os.path.join(self.data_dir, config.FILE_NAME_DICT['basic']))
        df_exchange = df.groupby('station_name')['line_name'].agg(
            {
                'num_line': 'nunique',
                'exchange_lines': lambda x: ','.join(sorted(list(set(x))))
            }
        ).reset_index()
        df_info = df[['station_name', 'station_lng', 'station_lat']].drop_duplicates()
        df_exchange = pd.merge(df_exchange, df_info, on='station_name', how='left')
        print('df_exchange shape is: ', df_exchange.shape)
        file_path = os.path.join(self.data_dir, config.FILE_NAME_DICT['exchange'])
        df_exchange.to_excel(file_path, index=False)
        return

    def find_adj_stations(self, limit_distance_list):
        df = pd.read_excel(os.path.join(self.data_dir, config.FILE_NAME_DICT['exchange']))
        adj_df_list = list(map(lambda x: self.spec_adj_stations(df, x), limit_distance_list))
        adj_dfs = adj_df_list[0]
        for adj_df in adj_df_list[1:]:
            adj_dfs = pd.merge(adj_dfs, adj_df, on='station_name', how='left')
        print('adj_dfs shape is: ', adj_dfs.shape)
        file_path = os.path.join(self.data_dir, config.FILE_NAME_DICT['adj'])
        adj_dfs.to_excel(file_path, index=False)
        return

    def spec_adj_stations(self, df, limit_distance=3):
        adj_stations = 'adjst_' + str(limit_distance) + 'km'
        num_adj_stations = 'num_' + adj_stations
        metro_dict = defaultdict(lambda: dict())
        for idx, row in df.iterrows():
            key = row['station_name']
            metro_dict[key]['position'] = {'lng': row['station_lng'], 'lat': row['station_lat']}
            metro_dict[key][adj_stations] = {}

        for main_key, main_val in metro_dict.items():
            for sub_key, sub_val in metro_dict.items():
                ms_dist = self._cal_distance(main_val['position'], sub_val['position'])
                if ms_dist > 0 and ms_dist <= limit_distance:  # exclude itself
                    metro_dict[main_key][adj_stations][sub_key] = ms_dist
            metro_dict[main_key][num_adj_stations] = len(metro_dict[main_key][adj_stations])

        metro_data = []
        for key, val in metro_dict.items():
            metro_data.append([key, val[num_adj_stations], val[adj_stations]])
        adj_df = pd.DataFrame(
            data=metro_data,
            columns=['station_name', num_adj_stations, adj_stations]
        )
        return adj_df

    def plot_stats(self):
        df_exchange = pd.read_excel(os.path.join(self.data_dir, config.FILE_NAME_DICT['exchange']))
        spec_col = 'num_line'
        title = 'distribution_of_exchange_lines'
        self._plot_val_cnt(df_exchange, spec_col, title, annotation=True)
        del df_exchange

        df_adj = pd.read_excel(os.path.join(self.data_dir, config.FILE_NAME_DICT['adj']))
        cols = df_adj.columns
        spec_cols = list(filter(lambda x: x.startswith('num_'), cols))
        for spec_col in spec_cols:
            limit_distance = spec_col.split('_')[-1]
            title = 'distribution_of_adjacent_stations_within_' + limit_distance
            annotation = True if int(limit_distance.strip('km')) <= 1 else False
            self._plot_val_cnt(df_adj, spec_col, title, annotation=annotation)
        del df_adj
        return

    def _cal_distance(self, position1, position2):
        # ref to: https://blog.csdn.net/vernice/article/details/46581361
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # 地球平均半径，单位为公里
            return round(c * r, 2)
        return haversine(position1['lng'], position1['lat'], position2['lng'], position2['lat'])

    def _plot_val_cnt(self, df, spec_col, title, annotation=False):
        df_stats = df[spec_col].value_counts().to_frame().reset_index()
        df_stats.columns = [spec_col, 'station_cnt']
        df_stats.sort_values(by=spec_col, ascending=True, inplace=True)
        x, y = list(zip(*df_stats[[spec_col, 'station_cnt']].values))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.bar(x, y)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        if annotation:
            for i in range(len(x)):
                ax.text(x[i], y[i] + 1, y[i])
        ax.set_xlabel(spec_col, fontsize=20)
        ax.set_ylabel('station_cnt', fontsize=20)
        ax.set_title(title, fontsize=20)
        img_path = os.path.join(self.data_dir, title + '.jpg')
        plt.savefig(img_path)
        plt.close()
        return


if __name__ == '__main__':

    metStaer = MetroStats('上海')
    metStaer.find_exchange_lines()
    metStaer.find_adj_stations([1, 3])
    metStaer.plot_stats()
