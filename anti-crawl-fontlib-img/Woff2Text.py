import os
import random
import requests
import json
from lxml import etree
from fontTools.ttLib import TTFont
from io import BytesIO
import time


class Woff2Text:
    
    def __init__(self, woff_target, margin_ratio=0.15):
        self.woff_target = woff_target
        self.margin_ratio = margin_ratio  # add space at edge (scale character)
        self.xml_file = self.woff_to_xml()
        self.text = self.xml_to_text()

    def woff_to_xml(self):
        dir_name, file_name_stem, _ = Woff2Text.get_dir_file_name(self.woff_target)
        xml_file = os.path.join(dir_name, file_name_stem + '.xml')
        font = TTFont(self.woff_target)
        font.saveXML(xml_file)
        return xml_file
    
    def xml_to_text(self):
        html = etree.parse(self.xml_file)
        graphs = html.xpath('.//TTGlyph')
        text = map(self.parseGraph, graphs)
        text = list(filter(None, text))  # some elements are None
        return dict(text)

    def parseGraph(self, graph):
        key = graph.xpath('./@name')[0]
        if key in ['glyph00000', 'x']:  # unuseful keys
            return None
        content = dict()
        xMin = int(graph.xpath('./@xMin')[0])
        xMax = int(graph.xpath('./@xMax')[0])
        yMin = int(graph.xpath('./@yMin')[0])
        yMax = int(graph.xpath('./@yMax')[0])

        width = xMax - xMin
        delta_width = int(width * self.margin_ratio)
        width += 2 * delta_width
        height = yMax - yMin
        delta_height = int(height * self.margin_ratio)
        height += 2 * delta_height

        content['width'] = width
        content['height'] = height
        
        contours = graph.xpath('.//contour')

        def parseContour(contour):
            nodes = contour.xpath('.//pt')
            x = map(lambda node: int(node.xpath('./@x')[0]) - xMin + delta_width, nodes)
            y = map(lambda node: int(node.xpath('./@y')[0]) - yMin + delta_height, nodes)
            return list(zip(x, y))

        data = list(map(parseContour, contours))
        content['data'] = data
        return key, content
    
    @staticmethod
    def get_dir_file_name(file_path):
        dir_name, file_name = os.path.split(file_path)
        file_name_stem, file_name_suffix = file_name.split('.')
        return dir_name, file_name_stem, file_name_suffix


class WoffUrl2Text(Woff2Text):

    # parse url in memory directly and skip file operation (save in local disk)
    def woff_to_xml(self):
        headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"
        }
        response = requests.get(url="https://" + self.woff_target, headers=headers)
        woff_file = BytesIO()
        for chunk in response.iter_content(100000):
            woff_file.write(chunk)
        font = TTFont(woff_file)
        del woff_file
        tmp_xml = BytesIO()
        font.saveXML(tmp_xml)
        xml_file = BytesIO(tmp_xml.getvalue())  # neccssary
        del tmp_xml
        return xml_file


if __name__ == '__main__':

    start = time.time()
    woff_file = Woff2Text('./files/demo.woff')
    text = woff_file.text
    with open('./files/demo.json', 'w') as fw:
        json.dump(text, fw)
    print('time cost in text generation from woff file is: ', time.time()-start)

    '''
    start = time.time()
    woff_file = WoffUrl2Text('urltowoff.woff')
    text = woff_file.text
    print('time cost in text generation from woff url is: ', time.time()-start)
    '''