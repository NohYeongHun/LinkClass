# AI 신호처리

## 인공신경망의 구조
1. Input Layer
- 데이터를 넣어주는 과정

2. Hidden Layer
- 데이터의 특성을 학습하는 과정.

3. Output Layer
- 분류나 회귀문제의 정답을 알려줌.


## 대표적인 인공 신경망
1. CNN
- Convolution 연산을 통해 결과 도출.
- 현재출력이 현재 입력만 영향.
- 일차원의 신호를 이차원으로 변환해서 데이터를 인풋값으로 넣음.

2. LSTM
- RNN의 일종
- 현재출력이 이전의 입력까지 고려함.
- 가지고 있는 데이터의 시간에 따라 변화하는 특성들을 인풋값으로 넣음


## Deep Learning Workflow
1. Create And Access Datasets
- 좋은데이터가 좋은 결과를 만든다.
- 연구목적
모델 개발에 압도적으로 많은 시간 소요가들어감. 

- 실제 산업
데이터개발에 많은 시간 소요가 들어감.


```
Datastore() : 대용량의 데이터셋을 가져 와서 처리를 쉽게 함.
사용 예시
a = audioDatastore(pwd, 'IncludeSubfolders', true...
, "LabelSource", "foldernames") 
```

- 데이터를 강화함으로써 대용량의 데이터를 만듬으로써 더 견고한 모델 생성 가능.

2. PREPROCESS AND TRANSFORM DATA

3. DEVELOP PREDICTIVE MODELS
- Design
- Train
- Optimize

4. ACCELRATE AND DEPLOY


## 신호처리 - 푸리에 트랜스폼. 