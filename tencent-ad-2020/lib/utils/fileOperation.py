#!/usr/bin/env python
# coding=utf-8
# @file  : utils
# @time  : 5/17/2020 7:43 PM
# @author: shishishu

import os
import json
import numpy as np
import pandas as pd

class FileOperation:

    @staticmethod
    def safe_mkdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return

    @staticmethod
    def get_dir_file_name(file_path):
        dir_name, file_name = os.path.split(file_path)
        file_name_split = file_name.split('.')
        # protect codes
        file_name_stem = file_name_split[0] if len(file_name_split) >= 1 else ""
        file_name_suffix = file_name_split[1] if len(file_name_split) >= 2 else ""
        return dir_name, file_name_stem, file_name_suffix

    @staticmethod
    def load_json(file_path):
        json_dict = json.load(open(file_path, 'r', encoding='utf-8'))
        return json_dict

    @staticmethod
    def save_json(json_obj, file_path):
        json.dump(json_obj, open(file_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        return

    @staticmethod
    def load_csv(file_path, sep=',', has_header=True):
        df = pd.read_csv(file_path, sep=sep) if has_header else pd.read_csv(file_path, sep=sep, header=None)
        return df

    @staticmethod
    def save_csv(df, file_path, sep=',', has_header=True):
        if has_header:
            df.to_csv(file_path, sep=sep, index=False, float_format='%.4f')
        else:
            df.to_csv(file_path, sep=sep, header=None, index=False, float_format='%.4f')
        return