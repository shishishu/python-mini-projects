## Reconstruct Chinese Fontlib in Anti-Crawling
#### Background:
- In the website, a small proportion of Chinese characters are designedly covered for anti-crawling and encoded in .woff file.
- Parse this .woff file could reconstruct Chinese Fontlib (key-char pairs) and restore the correct info as original context.
- No fixed mapping are required and it adapts to dynamic situations.

#### Dependence:
- System: windows 10
- Python: 3.6.5
- Tesseract: 4.00.00alpha [installation guidance](https://www.cnblogs.com/jianqingwang/p/6978724.html)
- pytesseract
- opencv
- lxml
- fontTools

#### General idea:
- Convert .woff file/url to .xml file
- **Contour info of Chinese character could be found in .xml file**, which is the key to reconstruct fontlib [ref_website](http://www.sohu.com/a/259331155_176628)

one example in .xml file:
```
    <TTGlyph name="unie055" xMin="0" yMin="-223" xMax="1988" yMax="1533">
      <contour>
        <pt x="688" y="1233" on="1"/>
        <pt x="1208" y="1233" on="1"/>
        <pt x="1208" y="1393" on="1"/>
        <pt x="634" y="1393" on="1"/>
        <pt x="634" y="1519" on="1"/>
        <pt x="1208" y="1519" on="1"/>
        <pt x="1208" y="1705" on="1"/>
        ...
      </contour>
      <instructions/>
    </TTGlyph>
```

image:

![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/de.png)

- Generate image based on contour info
- Recognize character via ocr method

#### Data preprocessing
```python
python Woff2Text.py
```
> Both .woff file or url are supported as scource materials

#### Main process 
```python
python main_chunk.py
```
- **3 different methods** are progressively used to generate image based on contour info of Chinese character, which could influence the ocr recognizatin result.
- In order to improve ocr effiency, images of several chareracters are combined together and then do ocr at one time
    - It is found the main time cost in ocr is the calling of ocr engine
    - It could **speed up by 3x** when 8 chars (chunk_size = 8) are recognized compared to 1 (chunk_size = 1) singly.
- Multiprocessing pool is applied to accelerate further

See live demo in [woff_text_image_char_flow.ipynb](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/woff_text_image_char_flow.ipynb)


Method | 1 | 2 | 3
---|---|---|---
Example 1 | ![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie055_m1.png)|
Prediction | Correct
Example 2 | ![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie60f_m1.png)|![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie60f_m2.png)
Prediction | Wrong | Correct
Example 3 |![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie3e1_m1.png)|![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie3e1_m2.png)|![image](https://github.com/shishishu/anti-crawl-fontlib-img/blob/master/images/unie3e1_m3.png)
Prediction | Wrong | Wrong | Correct

#### Results:
- text accuracy: > 90%
- time cost: 24s with chunk_size = 8

#### Online api
```python
python api.py
```

### More thinking
How to improve accuracy with higher effiency?
- Explore new image generation method when faced with dense lines or complex contours
- Train image-char pairs rather than ocr method to recognize character



