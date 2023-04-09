#!/usr/bin/env python
#coding=utf-8
# @file  : utils
# @time  : 5/24/2020 1:48 PM
# @author: shishishu

import json
import numpy as np
import pandas as pd
from datetime import datetime
from lib.utils.fileOperation import FileOperation

class ExtEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(ExtEncoder, self).default(obj)

def collect_log_content(log_dict, content, log_path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_dict[ts] = content
    FileOperation.save_json(log_dict, log_path)
    return

def collect_log_key_content(log_dict, key, content, log_path):
    log_dict[key] = content
    FileOperation.save_json(log_dict, log_path)
    return

def sample_df(df, target_frac):
    frac_list = [1.0 for _ in range(int(target_frac))]
    if target_frac - int(target_frac) > 0:
        frac_list.append(round(target_frac - int(target_frac), 2))
    df_out = pd.DataFrame()
    for frac in frac_list:
        tmp_df = df.sample(frac=frac, random_state=7)
        df_out = pd.concat([df_out, tmp_df], axis=0)
    return df_out

def sample_df_pipeline(df, target_domain, sample_strategy):
    df_new = pd.DataFrame()
    domain_vals = list(set(df[target_domain]))
    for domain_val in domain_vals:
        df_spec = df[df[target_domain] == domain_val]
        target_frac = sample_strategy.get(domain_val, 0)
        df_out = sample_df(df_spec, target_frac)
        df_new = pd.concat([df_new, df_out], axis=0)
    df_new = df_new.sample(frac=1.0, random_state=7)  # shuffle
    return df_new

def calculate_dist(df, target_domain):
    tmp = df[['user_id', target_domain]].groupby(target_domain).count().reset_index()
    tmp.columns = [target_domain, 'user_record']
    tmp['user_rate'] = tmp['user_record'].apply(lambda x: round(x / tmp['user_record'].sum(), 4))
    dist_dict = dict(tmp[[target_domain, 'user_rate']].values)
    return dist_dict

def calculate_delta_dist(stan_dist, pred_dist):
    delta_dist = dict()
    for key in stan_dist.keys():
        delta_dist[key] = round(pred_dist.get(key, 0) - stan_dist[key], 4)
    delta_dist_sort = sorted(delta_dist.items(), key=lambda x: x[0], reverse=False)
    return dict(delta_dist_sort)