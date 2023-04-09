#!/usr/bin/env python
#coding=utf-8
"""
Created on Wed, 1 May 2019
@author: Nano Zhou
"""

import urllib
import urllib.request
import base64
import json
import glob
from img_google_ocr import get_dir_file_name


def get_result_from_baidu(url, params):
    request = urllib.request.Request(url, params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read()
    content = json.loads(content)
    return content

def request_pipeline(file_path, url):
    with open(file_path, 'r') as fr:
        base64_data = fr.read().split(',')[1]
    params = {'image': base64_data}
    params = urllib.parse.urlencode(params)
    params = params.encode('utf-8')
    results = get_result_from_baidu(url, params)
    while 'error_msg' in results:
        results = get_result_from_baidu(url, params)  # continue to request when error meets
    _, file_name_stem, _ = get_dir_file_name(file_path)
    json_file_path = './input/baidu_api/' + file_name_stem + '.json'
    with open(json_file_path, 'w') as fw:
        json.dump(results, fw)


if __name__ == '__main__':
    access_token = 'abcdefg'  # get from baidu_access_token.py
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token=' + access_token

    file_names = glob.glob('./input/origin_base64/*.txt')

    for file_path in file_names:
        request_pipeline(file_path, url)