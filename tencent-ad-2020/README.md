### tencent-ad-2020

#### [Introduction](https://algo.qq.com/index.html?lang=en)
- Topic: estimation of the demographic attributes of Advertising audience
- Problem: use responsive behavior of users in the system to predict demographic features (age and gender)

#### Basic idea
- Do EDA (explore data analysis) to understand and clean data
- Generate item embedding from click sequence via word2vec method in NLP
- Generate user embedding based on item embedding
- Do classification (age: 10-class, gender: 2-class)

#### EDA
- AGE_DIST_DICT = {1.0: 0.0391, 2.0: 0.1659, 3.0: 0.2255, 4.0: 0.1673, 5.0: 0.1452, 6.0: 0.113, 7.0: 0.0741, 8.0: 0.0355, 9.0: 0.0216, 10.0: 0.0128}
- GENDER_DIST_DICT = {1.0: 0.6696, 2.0: 0.3304}
- See more details in eda/* and conf/config.py

#### Dataset
- clean data: lib/dataset/dataCleaner.py
    - input: click_log.csv, ad.csv
    - output: click_log_clean.csv
    - replace '\\N' with 0
    - click_times = min(x, click_times_99th)
    - random sample items if user_record >= user_record_99th

- create sequence: lib/dataset/createSeq.py
    - input: click_log_clean.csv
    - output: 'click_log_seq_' + ad_domain + '.txt'

- prepare samples: lib/dataset/dataMap.py
    - input: 'click_log_seq_' + ad_domain + '.txt'
    - output: 'tr/va_1/va_2/te_' + pred_domain + '.txt'

- submit/fuse results: lib/dataset/dataUtils.py
    
#### Model
- w2v: lib/model/w2v.py
    - input: click_log_seq_train.corpus, click_log_seq_test.corpus
    - output: w2v_embed_64.txt, w2v_embed_64.pkl, word.map

- xgb: lib/model/xgb.py, embedding of creative_id only 

- mlp: lib/model/mlp.py, embedding of creative_id only 

- rnn: lib/model/rnn.py, embedding of creative_id only 

- rnn2: lib/model/rnn2.py, embeddings of creative_id, advertiser_id

#### Results

|pred_model|eval_gender_precision|eval_age_precision
|---|---|---
|xgb|0.92|0.36
|mlp|NA|0.38
|rnn|0.94|0.42
|rnn2|NA|0.45
|fusion|0.945|0.46

#### More thinking
- Try more networks to generate user embedding based on item embeddings pre-trained by word2vec
- End-to-end: generate user embedding directly via Graph Network, e.g. node2vec