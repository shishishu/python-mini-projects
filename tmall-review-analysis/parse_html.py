#!/usr/bin/env python
#coding=utf-8
# @file  : parse_html
# @time  : 2/23/2020 4:00 PM
# @author: shishishu

import glob
import pandas as pd
import os
import argparse
from lxml import etree
from conf import config

parser = argparse.ArgumentParser()
parser.add_argument('--shop_name', type=str, required=True, help='shop name')  # 传递参数

def parse_html_files(html_file_path):
    dir_path, file_name = os.path.split(html_file_path)
    _, item_name = os.path.split(dir_path)  # 商品名
    page_id = file_name.split('.')[0]  # 评论页数

    parser = etree.HTMLParser(encoding='utf-8')
    html = etree.parse(html_file_path, parser=parser)

    nodes = html.xpath('//div[@class="tm-rate-content"]')

    output = []
    for node in nodes:
        try:
            comment = str(node.xpath('./div')[0].text).encode('ISO-8859-1').decode('gbk')
            if comment:
                output.append([item_name, page_id, comment])
        except:
            pass
    return output

def dump_data(html_files, output_file):
    big_data = []
    for html_file_path in html_files:
        output = parse_html_files(html_file_path)
        big_data.extend(output)

    item_name, page_id, comment = list(zip(*big_data))  # 一列拆多列
    df = pd.DataFrame(
        data={'item_name': item_name, 'page_id': page_id, 'comment': comment},
        columns=['item_name', 'page_id', 'comment']
    )
    print('df shape is: ', df.shape)
    df.sort_values(by=['item_name', 'page_id'], inplace=True)  # 按照item_name, page_id排序
    df.to_excel(output_file, index=False)
    return


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()
    shop_name = FLAGS.shop_name

    spec_data_dir = os.path.join(config.data_dir, shop_name)
    html_files = glob.iglob(os.path.join(spec_data_dir, '**', '0*.html'), recursive=True)  # 递归枚举当前文件夹下的文件
    output_file = os.path.join(spec_data_dir, 'raw_comment_' + shop_name + '.xlsx')

    dump_data(html_files, output_file)
    print('task is done...')


