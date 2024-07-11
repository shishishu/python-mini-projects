#!/usr/bin/env python
#coding=utf-8
# @file  : ImgFetcher
# @time  : 2/13/2022 6:52 PM
# @author: shishishu

import os
import urllib.request
import requests


class ImgFetch:

    def __init__(self, id, web_url, img_url_prefix, base_dir, *args, **kwargs):
        self.id = str(id)
        self.url = web_url
        self.img_url_prefix = img_url_prefix
        self.base_dir = base_dir
        self.index_dir = os.path.join(self.base_dir, 'index')
        self.html_dir = os.path.join(self.base_dir, 'html')
        self.img_dir = os.path.join(self.base_dir, 'img')
        self.img_info = {}
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

    def download_html(self):
        html = urllib.request.urlopen(self.url).read()
        self.html_path = os.path.join(self.html_dir, self.id + '.html')
        with open(self.html_path, 'wb') as fw:
            fw.write(html)
        return

    def parse_html(self): pass

    def download_image(self): pass

    def post_process(self): pass