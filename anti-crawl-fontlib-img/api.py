#!/usr/bin/env python
#coding=utf-8
"""
Created on Fri, 17 May 2019
@author: Nano Zhou
"""

from flask import Flask, request, jsonify
import traceback
import os
import json
import requests
from main_chunk import process


app = Flask(__name__)

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    try:
        query = request.get_json()['url']
        results = process(query, is_url=True, num_process=8, chunk_size=8)  # pass params as **kwargs
        """parsing .woff file to fontlib (key-char pairs)
        :param is_url: url is pending to parse if True, otherwise .woff file is passed
        :param num_process: number of processes in multiprocessing
        :param chunk_size: number of chars to be parsed by tesseract ocr at one time
            - larger chunk_size could save ocr parsing time, but also increases risk to return wrong results 
        """
        return jsonify(results)
    except: 
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':

    app.run(port=12345, debug=True)