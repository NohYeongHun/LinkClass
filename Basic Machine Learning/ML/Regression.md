# Multiple feature
[앤드류응강의]https://www.coursera.org/lecture/machine-learning/multiple-features-6Nj1q
1. 가격을 나타내는 여러 요소들을 벡터화한다.
```python
x(1) = vector(2104, 5, 1, 45) # 4차원 벡터, in-dimentional feature
x(2) = vector(1416, 3, 2, 40)
x(3) = vector(1534, 3, 2, 30)
```
2. Hypothesis:
- 표현식 : Θ(0) + Θ(1)*x(1) + Θ(2)*x(2) + Θ(3)*x(3) + Θ(4)*x(4)
3. Multivariate linear regression
- 종속 변수와 독립 변수 사이에 선형관계가 존재하는가?