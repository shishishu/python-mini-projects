#!/usr/bin/env python
#coding=utf-8
"""
Created on Sun, 5 May 2019
@author: Nano Zhou
"""

from flask import Flask, request, jsonify
import traceback
import base64
import torch
from userPred import * 
import os
import json
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            query = request.get_json()['image']
            base64_data = query.split(',')[1]
            img_data = base64.b64decode(base64_data)

            img_path = BytesIO()
            img_path.write(img_data)  # write in memory, skip saving image in local disk
            prediction = predict_image(model, img_path)
            img_path.seek(0)
            
            return jsonify({'predition': prediction})
        except: 
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model to use')


if __name__ == '__main__':
    model = ResNet(ResidualBlock)
    model.eval()  # set dropout and batch normalization layers to evaluation mode before running inference
    model.loadIfExist()
    app.run(port=12345, debug=True)