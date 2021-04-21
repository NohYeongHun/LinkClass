# 선형회귀 순서도
1. perch_full(데이터 불러오기)
- pd.read_csv().to_numpy()

2. train_input, test_input(훈련데이터, 실전데이터 분리)
- train_test_split()
- LinearRegression()
- fit(), score(), predict()

3. train_poly, test_poly(다항특성 만들기)
- PolynomialFeatures(degree=2), fit(), transform()
- LinearRegression(), fit(), score(), predict()

4. train_scaled, test_scaled(규제)
- StandScaler(), fit(), score(), predict()
- Ridge(alpha=1)
- fit(), score(), predict()
- Lasso(alpha=1),fit(), score), predict()

# KNN 다중 분류


# Predict / Proba
1. Predict
- 예측한 결과만

2. Proba
- 확률이 나옴