# Scikit-learn
[사이킷런]https://scikit-learn.org/stable/

# KNN-Regression
- 이산적인 값 (X)
- 연속적인 값 (O)

## k- 최근접 이웃 분류
- KNeighborsClassfier()는 일정 이웃 데이터 숫자로 결정.
- RadiusNeighborsClassifier()는 일정 거리내 데이터로 결정

## k- 최근접 이웃 회귀
- KNeighborsRegressor()는 일정 이웃 데이터값 평균으로 예측
- RadiusNeighborsRegressor()는 일정 이웃 거리값 평균으로 예측.

- 결정계수(R^2) = 1 - (타깃-예측)^2의 합 / (타깃-평균)^2의 합

### Overfitting(과대적합)과 Underfitting(과소적합)
- 모델 학습을 진행하면서 만나는 일반적인 문제.
- TestError의 변곡점이 Best Fit이다.
- 훈련데이터셋 평가점수 > 테스트데이터셋 평가 점수 (Overfitting)
- 훈련데이터셋 평가점수 < 테스트데이터셋 평가 점수 (Underfitting)
- BestFit지점을 정의해야한다.


# 농어의 무게를 예측하라.
1. 데이터 준비
2. ML모델 생성 - 모델결정, 훈련
3. 모델평가
4. 과정 : 길이 => ML Model => 연속적인 값 무게 예측
- 특성 하나만 사용함.
- 그래프 표현을 쉽게 하기 위함.

```python
# 회귀모델 생성 및 훈련(fit)
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)

# 회귀모델 평가
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                    weights='uniform')
knr.score(test_input, test_target)

# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```
### 하이퍼파라메터(이웃 개수)조정
- 이웃의 개수는 기본(default)가 5임.
- 이웃의 개수를 1로, 3으로, 42로 했을 때 훈련과 테스트 데이터 평가.


### KNN의 한계점
- 길이가 50cm인 아주 큰 농어
- 무게를 1033g으로 예측 했는데 실제 무게를 재보니 1500g
- KNN은 거리기반으로 데이터를 예측하기 때문에 데이터셋안의 범위에 의존적이다. 데이터 범위를 벗어난 것에 대해서 예측할 수 없다.

# 머신러닝 알고리즘과 응용 분야
## Machine Learning

1. Supervised Learning
- CLASSFICATION
- REGRESSION

2. UnSupervised Learning
- CLUSTERING




