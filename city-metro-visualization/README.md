## city-metro-visualization

### Project intro:
- 城市地铁分布可视化

### Main process:
- 数据调用（高德地图）
- 数据处理及分析
- 可视化
- python相关模块：
    - excel: pandas
    - plot: matplotlib, folium

### Pipeline
```python
python proc_data.py --cityname_zh <cityname_zh>
```
change "CITY_NAME" as <cityname_zh> in plot_map.ipynb, and run all cells in jupyter

### Example with '上海'
- 依次运行：
    - python proc_data.py --cityname_zh 上海
    - run plot_map.ipynb in jupyter

- 数据分析：
idx | desc | pic | comment
|---|---|---|---
1|x: 地铁站换乘线路数|![image](https://github.com/shishishu/python-mini-projects/blob/master/city-metro-visualization/data/%E4%B8%8A%E6%B5%B7/distribution_of_exchange_lines.jpg)|拥有4条换乘线的地铁站为：世纪大道，龙阳路
2|x: 地铁站1公里范围内地铁站数|![image]()|附近有4个的地铁站为：四川北路，四平路，大世界，新闸路，耀华路
3|x: 地铁站3公里范围内地铁站数|![image]()|附近不低于25个的地铁站为：南京西路，淮海中路，陕西南路，南京东路，新闸路，汉中路，自然博物馆，黄陂南路，曲阜路

- 可视化：
idx | desc | pic | html_file
1|地铁站线路图|![image]()|[]()
2|地铁站密度热力图|![image]()|[]()
