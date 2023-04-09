#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue, 30 Apr 2019
@author: Nano Zhou
"""

import urllib.request
import json


# get client_id (AK), client_secret (SK) for baidu ocr application
AK = 'abcde'  # need update with real AK
SK = 'ABCED'  # need update with real SK

host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + AK + '&client_secret=' + SK
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()

access_token = json.loads(content)['access_token']
print(access_token)