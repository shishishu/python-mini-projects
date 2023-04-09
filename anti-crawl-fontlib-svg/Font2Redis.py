#!/usr/bin/env python
#coding=utf-8

"""
Created on Fri, 24 May 2019
# convert standard .ttf(or .ttc) to .svg online
# extract (d, unicode) pairs from .svg and save this root dict into redis system
@author: Nano Zhou
"""

import redis
import configparser
import time
from fontTools.ttLib import TTFont
from parseFonts import processFonts


class Font2Redis:

    def __init__(self, font_file, config_file='./config.ini'):
        self.font_file = font_file  # type of font file: .ttf or .woff
        self.font_lib = self.parse_font()
        self.conf = configparser.ConfigParser()
        self.conf.read(config_file)
        self.r = Font2Redis.config_redis(self.conf)
    
    def parse_font(self):
        web_font_dict = processFonts(self.font_file)  # (gname, d)
        gname_ucode_map = dict()
        font = TTFont(self.font_file)
        cmap = font['cmap']
        for ucode, gname in cmap.getBestCmap().items():
            gname_ucode_map[gname] = ucode
        font_lib = dict()
        for gname, d in web_font_dict.items():
            font_lib[d] = gname_ucode_map.get(gname, gname)
        return font_lib  # (d, ucode/gname)
    
    @staticmethod
    def config_redis(conf):
        host = conf.get('redis_Yahei', 'host')
        port = conf.get('redis_Yahei', 'port')
        db = conf.get('redis_Yahei', 'db')
        password = conf.get('redis_Yahei', 'password')
        # return redis.Redis(host=host, port=port, db=db, password=password)
        return redis.Redis(host=host, port=port, db=db)

    def set_redis(self):
        for key, val in self.font_lib.items():
            self.r.set(key, val)
            if len(key) > 100:
                self.r.set(key[:30], val)  # used to increase matching rate
                self.r.set(key[-30:], val)
    
    def get_redis(self, key):
        response = self.r.get(key).decode('utf-8')  # convert byte to string
        if str.isdigit(response):
            return chr(int(response))  # convert unicode to character
        else:
            return response


if __name__ == '__main__':

    key = "M60 -1533H1988V-1377H1198V-73Q1198 221 890 223Q748 223 506 219Q496 141 472 25Q650 41 836 43Q1020 51 1020 -137V-1377H60V-1533Z"

    start = time.time()
    task = Font2Redis('./fonts/MicrosoftYahei.ttf')
    task.set_redis()
    print('time cost is font2redis is: ', time.time()-start)
    print(task.get_redis(key))

    '''
    conf = configparser.ConfigParser()
    conf.read('./config.ini')
    r = Font2Redis.config_redis(conf)
    print(r.get(key))
    print(chr(int(r.get(key))))
    '''