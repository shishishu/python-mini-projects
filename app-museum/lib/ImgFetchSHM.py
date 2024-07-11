#!/usr/bin/env python
#coding=utf-8
# @file  : ImgFetchSHM
# @time  : 2/13/2022 7:00 PM
# @author: shishishu

import os
import json
from lib.ImgFetch import ImgFetch
from lib.Utils import Utils


class ImgFetchSHM(ImgFetch):

    def parse_html(self):
        self.index_path = os.path.join(self.index_dir, self.id + '.index')
        cmd = """cat {} | grep relations | grep imgPath | awk -F 'relations = ' '{{print $2}}'  | sed -e 's/null/\"null\"/g' | awk -F "|" '{{print $1}}' > {}""".format(self.html_path, self.index_path)
        res = os.system(cmd)
        if res:
            raise Exception("parse html failed with id={}".format(self.id))

    def download_image(self):
        index_data = open(self.index_path, encoding='utf-8').readlines()[0].strip()
        d = json.loads(index_data)[0]
        d_entity = d['museumEntity']
        assert str(d['code']) == self.id, "id not matched in index, id={}, code={}".format(self.id, d['code'])
        self.img_info['id'] = self.id
        self.img_info['title'] = d_entity.get('name', '')
        self.img_info['img_dir'] = self.img_dir
        if 'imgPath' not in d_entity:
            raise Exception("parse imgPath failed with id={}".format(self.id))
        # img_url = os.path.join(self.img_url_prefix, d_entity['imgPath'])
        img_url = self.img_url_prefix + d_entity['imgPath']
        img_path = os.path.join(self.img_dir, self.id + '_o.jpg')
        Utils.download_image(img_url, img_path)
        self.img_info['origin_url'] = img_url
        self.img_info['origin_name'] = os.path.basename(img_path)
        self.img_info['origin_status'] = 1
        if 'picPath' in d_entity:
            img_url = self.img_url_prefix + d_entity['picPath']
            img_path = os.path.join(self.img_dir, self.id + '_t.jpg')
            Utils.download_image(img_url, img_path)
            self.img_info['thumbnail_url'] = img_url
            self.img_info['thumbnail_name'] = os.path.basename(img_path)
            self.img_info['thumbnail_status'] = 1

    def post_process(self):
        self.index_abbr_path = os.path.join(self.index_dir, self.id + '.index.abbr')
        if 'origin_status' in self.img_info and self.img_info['origin_status'] == 1:
            cmd = """md5sum {} | awk '{{print $1}}'""".format(os.path.join(self.img_info['img_dir'], self.img_info['origin_name']))
            self.img_info['origin_md5'] = os.popen(cmd).readlines()[0].strip('\n').strip('\\')
        if 'thumbnail_status' in self.img_info and self.img_info['thumbnail_status'] == 1:
            cmd = """md5sum {} | awk '{{print $1}}'""".format(os.path.join(self.img_info['img_dir'], self.img_info['thumbnail_name']))
            self.img_info['thumbnail_md5'] = os.popen(cmd).readlines()[0].strip('\n').strip('\\')
        json.dump(self.img_info, open(self.index_abbr_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    def run(self):
        self.download_html()
        self.parse_html()
        self.download_image()
        self.post_process()


if __name__ == '__main__':

    imgFetchSHM = ImgFetchSHM('CI00005593', 'https://www.shanghaimuseum.net/mu/frontend/pg/article/id/CI00005593', 'https://www.shanghaimuseum.net/mu/', './test/')
    imgFetchSHM.run()