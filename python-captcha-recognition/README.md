## Captcha recognition from scratch with Python, OCR
#### Background:
- Only raw captcha files (base64 format) are saved in local disk
- No label infomation provided for training samples
#### Dependence:
- System: windows 10
- Python: 3.6.5
- Tesseract: 4.00.00alpha [installation guidance](https://www.cnblogs.com/jianqingwang/p/6978724.html)
- pytesseract
- opencv

#### General idea:
- Generate labels for each captcha automatically
    -  use google ocr package (tesseract) to convert image to string
    -  use baidu ocr api [api-link](https://cloud.baidu.com/doc/OCR/OCR-API.html#.E6.8E.A5.E5.8F.A3.E8.83.BD.E5.8A.9B) to convert image to string
    -  decide to trust the result as real labels if recognized both by google and baidu, disgard it otherwise. (double check)
- Modelling on training data (captcha image with generated labels) as supervised learning

### Data preparation
#### Labelled by google ocr (tesseract)
```
python img_google_ocr.py
```
process pipeline:
- convert base64 file to image
- read image and remove interfering line
- adaptive threshold (similar to binarization) and apply morphological transformation
- split one captcha into 4 sub images and recognize with pytesseract

file pipeline example:

1.txt -> 1.jpg -> 1_0_a.jpg, 1_1_b.jpg, 1_2_c.jpg, 1_3_d.jpg

1_0_a.jpg:
- '1': file index from txt file
- '0': sub index after image spliting
- 'a': predicted string from pytesseract

#### Labelled by baidu ocr api
```
python baidu_access_token.py
```
- return access_token with client_id (AK), client_secret (SK) used
- AK, SK are provided by baidu ocr api

```
python img_baidu_api.py
```
- use access_token
- return json file with recognized results

### Data processing
```
python data_process.py
```
google_jpg_dict: {'1': [(img_path0, 'k'), (img_path1, 'c')], ...}
```python
def gen_jpg_dict(output_path, max_num=99999, dir_name='./input/google_ocr/'):
    jpg_dict = dict()
    img_names = glob.glob(dir_name + '/*.jpg')
    # sorted files: 1.jpg, 1_0_a.jpg, 1_1_b.jpg, 1_2_c.jpg, 1_3_d.jpg, 2.jpg...
    sorted_img_names = sorted(img_names, key=lambda x: int(get_dir_file_name(x)[1].split('_')[0]))  # sorted by file name
    for i in range(1, max_num + 1):
        tmp_list = []
        for j in range(1, 5):
            tmp_img_path = sorted_img_names[5 * (i-1) + j]
            decode_result = decode_jpg_name(tmp_img_path, i)
            if decode_result:
                tmp_list.append(decode_result)
        jpg_dict[str(i)] = tmp_list
    with open(output_path, 'w') as fw:
        json.dump(jpg_dict, fw)
    return jpg_dict
```
baidu_confirm_dict: {'1': 'ke', ...}
```python
def gen_confirm_dict(output_path, max_num=99999, dir_name='./input/baidu_api/'):
    word_confirm_dict = dict()
    json_names = glob.glob(dir_name + '/*.json')
    sorted_json_names = sorted(json_names, key=lambda x: int(get_dir_file_name(x)[1]))
    for i in range(1, max_num + 1):
        tmp_json_name = sorted_json_names[i-1]
        _, json_name_stem, _ = get_dir_file_name(tmp_json_name)
        assert json_name_stem == str(i), 'wrong matching...'
        with open(tmp_json_name, 'r') as fr:
            word_result = json.load(fr)['words_result']
            if word_result != []:
                word_confirm_dict[str(i)] = word_result[0]['words'].lower()
            else:
                word_confirm_dict[str(i)] = []
    with open(output_path, 'w') as fw:
        json.dump(word_confirm_dict, fw)
    return word_confirm_dict
```
correct_label_dict: {'1': [(img_path0, 'k')], ...}
```python
def gen_correct_label_dict(google_jpg_dict, baidu_confirm_dict, output_path, max_num=99999):
    correct_label_dict = dict()
    for i in range(1, max_num + 1):
        jpg_item = google_jpg_dict[str(i)]
        confirm_item = baidu_confirm_dict[str(i)]
        match_result = [elem for elem in jpg_item if elem[1] in confirm_item]
        if match_result != []:
            correct_label_dict[str(i)] = match_result
    with open(output_path, 'w') as fw:
        json.dump(correct_label_dict, fw)
    return correct_label_dict
```
generate training data based on correct_label_dict:
- X: dim = [384], read image data from img_path
- y: dim = [36], derive from recognized string and convert via onehot mapping (total 36 classes as chars are converted to lowercase, 0-9&a-z)
- save as .pkl file for modelling
> 100K captcha images (400K single chars) are used as raw input

> only 100K single chars are used in modelling later with 'correct' labels after double check


### Modelling and summary
```
python classifier.py
```
Setup simple MLPClassifier from sklearn.neural_network

#### Results:
- training accuarcy on single char: 0.928
- test accuarcy on single char: 0.924
- total accuarcy on golden set: 0.56
    - 100 capthas in golden set are labelled manually
    - correct prediction means 4 chars in one captcha are predicted correctly together
- time cost in loading mlp model: 2ms
- time cost in single captcha prediction: 2ms

![summary](https://github.com/shishishu/python-captcha-recognition/blob/master/output/summary.PNG)

### Online api
```
python api_demo.py
```
Realize with flask module and single response time is around 10ms (test with Postman)
![api_demo](https://github.com/shishishu/python-captcha-recognition/blob/master/output/api_demo.PNG)

### More thinking
How to increase captcha (with 4 chars) recognition accuracy (0.56) even though single char prediction accuracy is high (0.93)?
- In theory, captcha recognition accuracy should be equal to (single char accuracy)^4. In this case, it is 0.93^4 = 0.75
    - wrong labels may occur in training data (wrongly recognized both by google and baidu)
    - need more training examples to avoid overfitting
- Think more about image preprocessing
- End to end traing
    - consider generate captcha with Captcha module (labels known in the generation) as training dataset
    - train captcha as one example (no splitting) with deep learning 
