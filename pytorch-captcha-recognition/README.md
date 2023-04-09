## Captcha recognition with CNN (ResNet)
#### Dependence:
- System: windows 10
- GPU: RTX 2080
- Python: 3.6.5
- pytorch: 1.0.0-gpu
- opencv
- captcha

#### General idea:
- Generate captcha-label pairs with captcha module
- Train CNN (ResNet) model on generative training dataset
- Main reference: [CNN_captcha](https://github.com/braveryCHR/CNN_captcha)

### Data preparation/processing
```
python generate_captcha.py
- custom_captcha_image.py
- parameters.py
```
Instead of importing captcha.image in captcha module directly, new custom_captcha_image.py is setup (based on image.py)
- Captcha with target web styles could be generated easily by customizing some parameters
    - image size
    - font types and size
    - noisy line
    - noisy dots
- Avoid revising image.py (sys file) every time
    - location of image.py: Path to Anaconda3\Lib\site-packages\captcha

> 200K captcha images are generated for training

> 10K captcha images are generated for test (validation)


### Modelling and summary
```
python -m visdom.server  # run in cmd, avoid errors reported
python main.py
- model.py
- train.py
- parameters.py
- dataset.py
- Visdom.py
```
#### Models:
- setup ResNet model as main structure
- use softmax for multi-class classification (single-char level)
- treat 4 chars (one captcha content) as one example for accuracy calculation (uppercase and lowercase is recognized different)
- size of saved model is around 40 M
- test accuarcy on test dataset reaches 0.75 after 5 epochs
```
python userTest.py
```
#### Results:

- test accuarcy on golden_set (labelled manually on real captcha images):
    - 0.30, uppercase and lowercase is recognized different
    - 0.94, convert to lowercase both (it is the real application scenarios)

### Online api
```
python userPred.py
```
Some changes made for runtime prediction compared to userTest.py
- use test data directly rather than dataLoader
- set num_workers=0 in dataLoader when used in prediction: [In windows, DataLoader with num_workers > 0 is extremely slow (50 times slower)](https://github.com/pytorch/pytorch/issues/12831)

![prediction](https://github.com/shishishu/pytorch-captcha-recognition/blob/master/output/prediction.PNG)
```
python api_demo.py
```
Implement with flask module and single response time is around 60 ms (test with Postman)
![api_demo](https://github.com/shishishu/pytorch-captcha-recognition/blob/master/output/api_demo.PNG)

### More thinking
Why is it that current classfication result is much better than [python-captcha-recognition](https://github.com/shishishu/python-captcha-recognition)? 
- Labels are always correct in custom_captcha_image rather than ocr methods
- Uppercase and lowercase is recognized different in current training process (total 62 classes) while all the letters are converted to lowercase and then do classification (total 36 classes) before
- Num of training examples is much higher and deep learning model (ResNet here) is more suitable
- End-to-end training: raw captcha iamge -> labels, no image preprocessing involved
- It is scalable to different web styles (changes in captcha generation is required only)
