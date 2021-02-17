## city-metro-visualization

### Project intro:
- 城市地铁分布可视化

### Main process:
- 数据调用（高德地图）
- 数据处理及分析
- 可视化
- python相关模块：
    - url: requests
    - excel: pandas
    - plot: matplotlib, folium

### Pipeline
```bash
python proc_data.py --cityname_zh <cityname_zh>
```
change "CITY_NAME" as <cityname_zh> in plot_map.ipynb, and run all cells in jupyter

### Example with '上海'
- 运行代码
    - python proc_data.py --cityname_zh 上海
     - run plot_map.ipynb in jupyter

- 数据分析

idx | desc | pic | comment
|---|---|---|---
1|x: 地铁站换乘线路数|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/distribution_of_exchange_lines.jpg)|拥有4条换乘线的地铁站为：<ul><li>世纪大道</li><li>龙阳路</li></ul>
2|x: 地铁站1公里范围内地铁站数|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/distribution_of_adjacent_stations_within_1km.jpg)|附近有4个的地铁站为：<ul><li>四川北路</li><li>四平路</li><li>大世界</li><li>新闸路</li><li>耀华路</li></ul>
3|x: 地铁站3公里范围内地铁站数|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/distribution_of_adjacent_stations_within_3km.jpg)|附近不低于25个的地铁站为：<ul><li>南京西路</li><li>淮海中路</li><li>陕西南路</li><li>南京东路</li><li>新闸路</li><li>汉中路</li><li>黄陂南路</li><li>曲阜路</li><li>自然博物馆</li></ul>

- 可视化

idx | desc | pic | html_file
|---|---|---|---
1|地铁站线路图|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/sh01.PNG)|[citymap_cluster.html](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/citymap_cluster.html)
2|地铁站密度热力图|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/sh02.PNG)|[citymap_heatmap.html](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/citymap_heatmap.html)
