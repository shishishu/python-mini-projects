## Anti-crawling with Unique d-path Feature in SVG

#### Dependence:
- System: windows 10
- Python: 3.6.5
- fontTools

#### General idea:
- Convert TTF file used in targeted website(e.g. MicrosoftYahei.ttf) to a series of SVG files 
- Extract (d-path, unicode) pairs into Redis system
- Convert WOFF file to SVG and retrieve unicode in Redis based on d-path

one example in transformed SVG file:
```
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -2048 2048 2048">
	<path fill="#000000" d="M386 261H230V-867H946V-1027H76V-1155H946V-1311H1102V-1155H1668Q1564 -1219 1466 -1275L1538 -1369Q1672 -1297 1782 -1231L1724 -1155H1980V-1027H1102V-867H1818V13Q1818 241 1598 241Q1462 241 1292 239Q1284 185 1262 85Q1424 97 1550 97Q1662 97 1662 -11V-135H1102V239H946V-135H386V261ZM1270 -1699H1414V-1531H2006V-1403H1414V-1245H1270V-1403H778V-1245H634V-1403H48V-1531H634V-1699H778V-1531H1270V-1699ZM1662 -743H1102V-559H1662V-743ZM1102 -255H1662V-439H1102V-255ZM386 -559H946V-743H386V-559ZM386 -255H946V-439H386V-255Z"/>
</svg>
```

#### Main process
```
python parseFonts.py
```
> Generates a set of SVG glyph files from one or more fonts (TTF, WOFF)

> revise codes based on [Scripts and sample files for making OpenType-SVG fonts](https://github.com/adobe-type-tools/opentype-svg)

```
python Font2Redis.py
```
- config redis (set configuration in config.ini)
```
def config_redis(conf):
    host = conf.get('redis_Yahei', 'host')
    port = conf.get('redis_Yahei', 'port')
    db = conf.get('redis_Yahei', 'db')
    password = conf.get('redis_Yahei', 'password')
    return redis.Redis(host=host, port=port, db=db, password=password)
```
- parse font 
```
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
```
- save d-path into redis (d[:30] and d[-30:] is also used as tiny variation in some characters)
```
def set_redis(self):
    for key, val in self.font_lib.items():
        self.r.set(key, val)
        if len(key) > 100:
            self.r.set(key[:30], val)  # used to increase matching rate
            self.r.set(key[-30:], val)
```
- get unicode based on d-path
```
def get_redis(self, key):
    response = self.r.get(key).decode('utf-8')  # convert byte to string
    if str.isdigit(response):
        return chr(int(response))  # convert unicode to character
    else:
        return response
```

#### Online api
```python
python api_font.py
```
- convert WOFF to SVG
- retrieve unicode based on d-path
```
def font_to_dict(self):
    parse_results = dict()
    web_font_dict = processFonts(self.url_to_file())
    for key, d in web_font_dict.items():
        response = self.r.get(d)
        char = Woff2Dict.convert_to_char(response)  # default setting (1st choice)
        if char == None and len(d) > 100:
            response_front = self.r.get(d[:30])  # use first 30 chars in d for matching (2nd choice)
            char = Woff2Dict.convert_to_char(response_front)
            if char == None:
                response_back = self.r.get(d[-30:])  # use last 30 chars in d for matching (3rd choice)
                char = Woff2Dict.convert_to_char(response_back)
                if char == None:
                    char = 'UNK'
        parse_results[key] = char
    return parse_results
```

#### Results:
- text accuracy: over 99% (almost 100%)
- time cost: within 1s, 20x faster than previous [OCR method](https://github.com/shishishu/anti-crawl-fontlib-img)