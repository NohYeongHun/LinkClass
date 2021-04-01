# Pandas
1. 데이터 분석에 자주 사용하는 테이블 형태를 라룰 수 있는 라이브러리
2. Series, DataFrame, Panel 지원
3. Series
- 리스트를 원소로 생성하는 1차원 자료구조
- 원소 인덱스는 0부터 시작하는 정수를 기본으로 사용
- 리스트 생성
- pd.Series() 사용하여 기존 리스트를 Series로 생성
4. DataFrame
- 행과 열이 있는 테이블 형태의 2차원 자료구조
- pd.DataFrame() 사용하여 딕셔너리를 DataFrame으로 생성.

# 데이터 시각화
1. 데이터를 그래프 표현 직관적ㅇ 이해 가능
- 데이터 변화의 흐름을 한 눈에 파악
- 현재 상황 이해가 쉬움
- 미래의 변화 추이까지 예측 가능
- 데이터의 여러 속성들 간 상관관계에 대한 통찰 제공
- 데이터의 통계적인 속성도 한눈에 알 수 있음.

2. matplotlib 라이브러리
- 파이썬에서 데이터 시각화 사용
- 2D 형태 그래프, 이미지 그릴 때 사용
- 과학 컴퓨팅 연구 분야나 인공지능 분야에서 많이 활용.
- matplotlib 홈페이지 : https://matplotlib.org/
- import matplotlib.pyplot as plt

# 기본 그래프 그리기
1. plt.plot()
![](2021-04-01-10-40-56.png)

2. 타이틀, 마커, 색상, 범례
![](2021-04-01-10-52-12.png)

3. 막대 그래프
- 두 개의 바를 상하로 배치한 차트
- 바 차트(그래프)에 사용할 y축 데이터로 y1, y2 준비
- y2에 대한 바 차트를 y1 위에 배치
- 데이터 준비
![](2021-04-01-11-00-53.png)

4. 기온변화 그래프로 나타내기
![](2021-04-01-11-17-50.png)

5. 데이터 분석 : 히스토그램
- 자료의 분포 상태를 직사각형 모양의 막대 그래프로 나타낸 것.
- 데이터의 빈도에 따라 높이가 결정됨.
- plt.hist() 사용

```python
list_hist = [1,1,2,2,3,3,4,4,4,5,5,5,5]
plt.hist(list_hist)
plt.show()
```
## 히스토그램 연습하기
1. 연습용 주사위 시뮬레이션
2. 임의의 수를 뽑는 랜덤 함수 사용
3. 주사위 시뮬레이션 진행 과정
- 주사위를 굴린다. => random 모듈의 randint() 사용
- 나온 결과를 기록한다.
- 1)~2) 과정을 n 번 반복한다.
- 주사위의 눈이 나온 횟수를 히스토그램으로 그린다.
```python
import random
print(random.randint(1,6))

dice = []
for i in range(5) :
    dice.append(random.randint(1,6))
print(dice)
plt.xlim(1,6)
plt.hist(dice, bins=6)
plt.show()

dice = []
for i in range(100):
    dice.append(random.randint(1,6))
print(dice)

plt.hist(dice, bins=6)
plt.show()

dice = []
for i in range(1000):
    dice.append(random.randint(1,6))
print(dice)

plt.hist(dice, bins=6)
plt.show()

dice = []
for i in range(10000):
    dice.append(random.randint(1,6))
print(dice)

plt.hist(dice, bins=6)
plt.show()

dice = []
for i in range(100000):
    dice.append(random.randint(1,6))
print(dice)

plt.hist(dice, bins=6)
plt.show()

dice = []
for i in range(1000000):
    dice.append(random.randint(1,6))
print(dice)

plt.hist(dice, bins=6)
plt.show()

점차 빈도그래프가 정사각형의 모양을 가진다.
```
# Pandas , matplotlib 사용법
```python
import pandas as pd
pd.__version__

data = [10,11,12,13,14,15]
data_test = pd.Series(data)
data_test

Out[]:
0    10
1    11
2    12
3    13
4    14
5    15
from matplotlib import pyplot as plt
f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data:
    if row[-1] != '' and row[-2] != '' :
        if 1983 <= int(row[0].split('-')[0]) :
            if row[0].split('-')[1] == '04' and row[0].split('-')[2] == '01':
                high.append(float(row[-1]))
                low.append(float(row[-2]))

plt.rc('font', family='Malgun Gothic')
plt.title('4월 1일의 기온 변화 그래프')
plt.rcParams['axes.unicode_minus']=False
plt.plot(high,'hotpink')
plt.plot(low, 'skyblue')
plt.show()

f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data:
    if row[-1] != '' and row[-2] != '' :
        if 1983 <= int(row[0].split('-')[0]) :
            if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '01':
                high.append(float(row[-1]))
                low.append(float(row[-2]))

plt.rc('font', family='Malgun Gothic')
plt.title('2월 1일의 기온 변화 그래프')
plt.rcParams['axes.unicode_minus']=False
plt.plot(high,'hotpink')
plt.plot(low, 'skyblue')
plt.show()
```
