#!/usr/bin/env python
#coding=utf-8
"""
Created on Mon, 6 May 2019
@author: Nano Zhou
"""

from flask import Flask, request, jsonify
import traceback
import base64
import os
import json
from sklearn.externals import joblib
from classifier import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            query = request.get_json()['image']
            base64_data = query.split(',')[1]
            img_data = base64.b64decode(base64_data)
            test_data = cvt_img_data(img_data, file_type='b64')  # use img_data directly, skip save img and load again
            prediction = clf_predict(model, test_data)
            return jsonify({'predition': prediction})
        except: 
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model to use')


if __name__ == '__main__':

    model = joblib.load(open('./output/mlp.model', 'rb'))

    app.run(port=12345, debug=True)