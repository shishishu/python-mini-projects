#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 24 May 2019
@author: Nano Zhou
"""

from flask import Flask, request, jsonify
import traceback
import os
import json
import requests
import configparser
from io import BytesIO
from Font2Redis import Font2Redis
from parseFonts import processFonts

class Woff2Dict:

    def __init__(self, woff_url, r):
        self.woff_url = woff_url
        self.r = r
        self.parse_result = self.font_to_dict()

    def url_to_file(self):
        headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"
        }
        response = requests.get(url="https://" + self.woff_url, headers=headers)
        woff_file = BytesIO()
        for chunk in response.iter_content(100000):
            woff_file.write(chunk)
        return woff_file
    
    def font_to_dict(self):
        parse_results = dict()
        web_font_dict = processFonts(self.url_to_file())
        for key, d in web_font_dict.items():
            response = self.r.get(d)
            char = Woff2Dict.convert_to_char(response)  # default setting (1st choice)
            if char == None and len(d) > 100:
                response_front = self.r.get(d[:30])  # use first 30 chars in d for matching (2nd choice)
                char = Woff2Dict.convert_to_char(response_front)
                if char == None:
                    response_back = self.r.get(d[-30:])  # use last 30 chars in d for matching (3rd choice)
                    char = Woff2Dict.convert_to_char(response_back)
                    if char == None:
                        char = 'UNK'
            parse_results[key] = char
        return parse_results
    
    @staticmethod
    def convert_to_char(response):
        if response != None:
            char = response.decode('utf-8')
            if str.isdigit(char):
                char = chr(int(char))
        else:
            char = None
        return char


app = Flask(__name__)

@app.route('/font', methods=['POST'])
def parse():
    try:
        woff_rul = request.get_json()['woff_url']
        parser = Woff2Dict(woff_rul, r)
        results = parser.font_to_dict()
        return jsonify(results)
    except: 
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':

    conf = configparser.ConfigParser()
    conf.read('./config.ini')
    r = Font2Redis.config_redis(conf)

    app.run(host='0.0.0.0', port=54321, debug=True)
