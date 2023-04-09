#!/usr/bin/env python

# Copyright 2016 Adobe. All rights reserved.

"""
Generates a set of SVG glyph files from one or more fonts and hex colors
for each of them. The fonts' format can be either OpenType, TrueType, WOFF,
or WOFF2.
"""


import os
import re
import sys
import time
import requests
from io import BytesIO

import util.check_fonttools  # pylint: disable=unused-import

from fontTools import ttLib
from fontTools.pens.basePen import BasePen
from fontTools.pens.transformPen import TransformPen


class SVGPen(BasePen):

    def __init__(self, glyphSet):
        BasePen.__init__(self, glyphSet)
        self.d = u''
        self._lastX = self._lastY = None

    def _moveTo(self, pt):
        ptx, pty = self._isInt(pt)
        self.d += u'M{} {}'.format(ptx, pty)
        self._lastX, self._lastY = pt

    def _lineTo(self, pt):
        ptx, pty = self._isInt(pt)
        if (ptx, pty) == (self._lastX, self._lastY):
            return
        elif ptx == self._lastX:
            self.d += u'V{}'.format(pty)
        elif pty == self._lastY:
            self.d += u'H{}'.format(ptx)
        else:
            self.d += u'L{} {}'.format(ptx, pty)
        self._lastX, self._lastY = pt

    def _curveToOne(self, pt1, pt2, pt3):
        pt1x, pt1y = self._isInt(pt1)
        pt2x, pt2y = self._isInt(pt2)
        pt3x, pt3y = self._isInt(pt3)
        self.d += u'C{} {} {} {} {} {}'.format(pt1x, pt1y, pt2x, pt2y,
                                               pt3x, pt3y)
        self._lastX, self._lastY = pt3

    def _qCurveToOne(self, pt1, pt2):
        pt1x, pt1y = self._isInt(pt1)
        pt2x, pt2y = self._isInt(pt2)
        self.d += u'Q{} {} {} {}'.format(pt1x, pt1y, pt2x, pt2y)
        self._lastX, self._lastY = pt2

    def _closePath(self):
        self.d += u'Z'
        self._lastX = self._lastY = None

    def _endPath(self):
        self._closePath()

    @staticmethod
    def _isInt(tup):
        return [int(flt) if (flt).is_integer() else flt for flt in tup]


def processFonts(font_path):

    # Load the fonts and collect their glyph sets
    font = ttLib.TTFont(font_path)
    gSet = font.getGlyphSet()
    glyphNamesList = gSet.keys()
    font.close()
    
    # Confirm that there's something to process
    if not glyphNamesList:
        print("The fonts and options provided can't produce any SVG files.",
              file=sys.stdout)
        return

    web_font_dict = dict()
    # Generate the SVGs
    for gName in glyphNamesList:
        
        pen = SVGPen(gSet)
        tpen = TransformPen(pen, (1.0, 0.0, 0.0, -1.0, 0.0, 0.0))
        glyph = gSet[gName]
        glyph.draw(tpen)
        d = pen.d
        # Skip glyphs with no contours
        if not len(d):
            continue

        web_font_dict[gName] = d  # (gname, d)
    
    font.close()

    return web_font_dict


if __name__ == "__main__":
    
    start = time.time()
    results = processFonts('./files/qidian.woff')
    print('time cost is: ', time.time()-start)
    # print(results)
    
    '''
    headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"
        }
    response = requests.get(url="https://" + 's3plus.meituan.net/v1/mss_73a511b8f91f43d0bdae92584ea6330b/font/53cfe63b.woff', headers=headers)
    woff_file = BytesIO()
    for chunk in response.iter_content(100000):
        woff_file.write(chunk)
    results = font2svg(woff_file)
    print(results)
    '''
    