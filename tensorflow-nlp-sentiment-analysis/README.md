## Multi  Aspect Sentiment Analysis on User Reviews
### Background:
- Project: [Fine-grained Sentiment Analysis of User Reviews of AI Challenger 2018](https://challenger.ai/competition/fsauor2018)
- Original dataset could be [downloaded here](https://pan.baidu.com/s/1a3ft6nuKpP8baqwoprmzhA ) with fetch code "fiag"

### Dependence:
- system: windows 10
- python: 3.6.5
- tensorflow-gpu: 1.1.10

### General idea:
- Path 1: single-task-learning
    - Train each aspect as multi-class classification
    - Number of independent models equals to total aspects (20 in this case)
- Path 2: multi-task-learning
    - Only one model required as joint learning
	    - Avoid high time cost in the training
	    - Avoid large memory space of final model
	    - Decrease risk of overfitting via features sharing
    - Hard to train as too many parameters in network

### Data preprocess

#### word2vec
```
python w2v.py \
    --num_porcess 8 \
    --embedding_size 100 \
    --map_file word.map
```
- input: corpus
- output: 
    - w2v.model
    - w2v.pkl
    - w2v.txt

Check word2vec training via synonym finding:

![image](https://github.com/shishishu/tensorflow-nlp-sentiment-analysis/blob/master/images/w2v.png)

#### data pipeline
```
python data_pipeline.py \
    --num_process 8 \
    --task_type train \
    --save_seg True
```
File structure:

    ├── input
        ├── raw_inputs
            ├── train
                ├── tr.csv
        ├── segments
            ├── train
                ├── tr_seg.txt
        ├── encodes
            ├── train
                ├── tr_enc.txt

File pipeline example:

- record in tr.csv: "第三次参加大众点评网霸王餐的活动。这家店给人整体感觉一般。..."
- record in tr_seg.txt: "第三次 参加 大众 点评 网 霸王餐 活动 这家 店 人 整体 一般 ..."
- record in tr_enc.txt: [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 70 7198 7096 44624 12 13219 82 80 44 177 639 2359 13219 84 80 44 59530 6685 297 50 330 13219 3177 900 39 70 999 441 841 810 2175 2142 1846 1552 3201 7 108 24902 59531 13823 7 108 7287 338 5291 788 115 873 5291 6398 841 1082 5291 602 7 18680 9 59532 7734 59533 5291 124 26 1584 1170 5291 126 639 84 19 17714 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    - y-onehot: [:80]
    - sen_len: [80]
    - word_enc: [81:], padding with 0 as max_sen_len

### Single-task-learning

#### LSTM and its variants
```
python tf_lstm_variants.py \
    --embedding_size 100 \
    --num_lstm_units 100 \
    --max_sen_len 240 \
    --batch_size 500 \
    --learning_rate 0.001 \
    --num_epoch 50 \
    --skip_step 50 \
    --skip_epoch 1 \
    --keep_pro 0.8 \
    --num_deep_rnn 2 \
    --aspect_embedding_size 100 \
    --aspect_id 15 \
    --model_type 'lstm'
```
- Basic model structure

![image](https://github.com/shishishu/tensorflow-nlp-sentiment-analysis/blob/master/images/lstm.PNG)

- Result comparison on valid dataset

Model | macro_f1_score | accuracy_score
--- | --- | ---
lstm | 0.6879 | 0.7694
bilstm | 0.6849 | 0.7711
deeplstm | 0.6552 | 0.7645
[aeatlstm](https://aclweb.org/anthology/D16-1058) | 0.6930 | 0.7741

#### Multi-head attention (ref to [Encoder Structure in Transformers](https://arxiv.org/abs/1706.03762))
```
python tf_multiatten_estimator.py \
    --embedding_size 100 \
    --num_lstm_units 100 \
    --max_sen_len 210 \
    --bilstm_depth 1 \
    --d_model 200 \
    --num_atten_head 4 \
    --atten_mask True \
    --layer_norm True \
    --pool_method 'no_pool' \
    --pool_topk 3 \
    --inter_units 32 \
    --num_thread 8 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --log_steps 1 \
    --keep_prob 0.8 \
    --aspect_id 15 \
    --task_type 'train' \
    --dt_dir '20190529' \
    --clear_existing_model False
```
- Model structure

![image](https://github.com/shishishu/tensorflow-nlp-sentiment-analysis/blob/master/images/multiatten.PNG)

- Result: accuracy_score: 0.7575

##### Tensorflow serving with Docker
- export model
```
python tf_multiatten_estimator.py --task_type 'export'
```
- run in docker
```
docker run -t --rm -p 8501:8501 -v "/D/python/masa_meituan_all/output/matt/aspect_15/server/20190529_1:/models/matt_mq_seq" -e MODEL_NAME=matt_mq_seq tensorflow/serving
```
- predict
```
curl -H "Content-type: application/json" -X POST -d '{"instances": [{"sen_encode": [16158, 14, 207690, 11, 17787, 121401, 18621, 1454, 367, 9559, 427, 35, 15, 15, 2, 219170, 18381, 2361, 473, 7048, 259, 30, 3, 7, 655, 956, 98, 5263, 8432, 60, 1143, 10643, 5, 1, 10336, 567, 4, 547, 26, 175, 3, 22643, 24, 950, 3400, 7, 1036, 18621, 2, 398, 7612, 427, 27, 203, 15668, 23, 231, 1313, 3, 6, 1938, 184, 1, 304, 207690, 12, 3976, 4566, 6887, 9, 1471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "sen_len": [71]}]}' "http://localhost:8501/v1/models/matt_mq_seq:predict"
```
- result

![image](https://github.com/shishishu/tensorflow-nlp-sentiment-analysis/blob/master/images/docker.PNG)

### Multi-task-learning (idea provided only)
![image](https://github.com/shishishu/tensorflow-nlp-sentiment-analysis/blob/master/images/mtl.PNG)

### Reference
1. [AI Challenger 2018：细粒度用户评论情感分析冠军思路总结](https://tech.meituan.com/2019/01/25/ai-challenger-2018.html)
2. [AI Challenger2018情感分析赛道亚军PPT分享](https://cloud.tencent.com/developer/news/378276)
3. [Attention-based LSTM for Aspect-level Sentiment Classification](https://aclweb.org/anthology/D16-1058)
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
5. [bert-multi-gpu](https://github.com/HaoyuHu/bert-multi-gpu)
6. [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker)