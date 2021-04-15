```python
2021.03.25
========
f = open('gugu.txt', 'r')
while True:
    line=f.readline()
    if not line:
        break
    print(line)
f.close()
========
f = open("./seoul.csv", 'r')
num = 0
while True:
    line=f.readline()
    if not line:
        break
    num += 1
    print(line)
f.close()
print(num)
=========
#입력받기 코드
nat01 = input("국적 코드 입력 : ")
gen01 = input("성별 입력 : ")
noPassengerp01 = input("입국 객수 입력 :")
noPassengerb01 = input("전년 동기 입력 :")
=========
#입력 결과 확인용
print(nat01)
print(gen01)
print(noPassengerp01)
print(noPassengerb01)
=======
p1 = [nat01, gen01, noPassengerp01, noPassengerb01]
p1
===========
p1, passenger 확인
===========
pTot =0
bTot =0
passenger = []
for i in range(0, 6):
	nat01 = input("국적코드 입력 : ")
	gen01 = input("성별입력 : ")
	noPassenger01 =  input("입국 갯수 입력 : ")
	noPassenger02 = input("전년 동기 입력 : ")
	p1 = [nat01, gen01, noPassengerp01, noPassengerb01]
	passenger.insert(i,p1)
	pTot += int(noPassengerp01)
	bTot += int(noPassengerb01)
==========
p1 = passenger
f = open(".\passenger.txt", 'w')
for i in range(0,6):
    data = p1[i][0]+" "+p1[i][1] +" "+str(p1[i][2]) +" "+str(p1[i][3])+"\n"
    f.write(data)
f.close()
==========
print("  국적코드   성별       입국객수     전년동기")
print('-'*50)
for i in range(6):
    print("%-12s  %-7s %10s   %10s"%(passenger[i][0], passenger[i][1], passenger[i][2], passenger[i][3]))
    print('-'*50)
print('입국객수 총 합계    %d'%pTot)
print('전년동기 총 합계    %d'%bTot)
=============
f = open("./travel.csv", 'r')
while True:
    line=f.readline()
    if not line:
        break
    print(line)
f.close()
========
f = open("./travel.csv", 'r')
tra = []
tra2 = []
i = 0
fbTot = 0
fpTot = 0
while True:
    line=f.readline()
    tra.insert(i, line)
    if not line:
        break
#    print(line)
    tra2.insert(i, tra[i].split(","))
    print("%s, %s, %s, %s"%(tra2[i][0], tra2[i][1], tra2[i][2], tra2[i][3]))
    fpTot += int(tra2[i][2])
    fbTot += int(tra2[i][3])
    i+=1
f.close()
print(fpTot)
print(fbTot)
=============
print("  국적코드   성별       입국객수     전년동기")
print('-'*50)
for i in range(6):
    print("%-12s%-5s   %10d    %10d"%(tra2[i][0], tra2[i][1], int(tra2[i][2]), int(tra2[i][3])))
    print('-'*50)
print('입국객수 총 합계    %d'%fpTot)
print('전년동기 총 합계    %d'%fbTot)
========
파일 출력 코드 작성
========
def sum1(a,b):
    x = a+b
    return x
========
def sum2(*args):
    x=0
    for i in args:
        x += i
    return x
==========
a = 5
b = 3
sum1(a, b)
==========
a = 5
b = 3
sum1(a, b)
========
sum1(32767, 32767)
=========
내장 함수 연습
===========
import numpy as np
=========
np.__version__
=========
ar1 = np.array([1,2,3,4,5])
ar1
=========
ar2 = np.array([[10, 20, 30], [40, 50, 60]])
ar2
======
ar3 = np.arange(1, 11, 2)
ar3
==========
ar4 = np.array([1,2,3,4,5,6]).reshape((3,2))
ar4
========
ar4 = np.array(ar4).reshape((2,3))
ar4
=======
ar5 = np.zeros((2,3))
ar5
====
ar6 = ar2[0:2, 0:2]
ar6
=====
ar7 = ar2[0, :]
ar7
====
ar8 = ar1 + 10
ar8
=====
ar1 + ar8
=====
ar8 - ar1
==
ar1 * 2
===
ar1 / 2
======
ar9 = np.dot(ar2, ar4)
ar9
=====
a4 = np.array(ar4).reshape((3,2))
a4
=====
ar9 = np.dot(ar2, a4)
ar9
=======
import csv 
====
f = open('seoul.csv', 'r') 
data = csv.reader(f, delimiter=',') 
for row in data :
    print(row)

f.close() 

===
f =open('seoul.csv')
data = csv.reader(f)
header =next(data)   #파일에서 헤더(전체 데이터의 타이틀) 읽기
print(header)        
f.close()

==
f =open('seoul.csv')
data = csv.reader(f)
header =next(data)
for row in data :
    print(row)
f.close()

==
f =open('seoul.csv')
data = csv.reader(f)
header =next(data)
for row in data :
    row[-1] = float(row[-1]) # 최고 기온을 실수로 변형
    print(row)

==
max_temp =-999   # 최고 기온 값을 저장할 변수
max_date =''       # 최고 기온이 가장 높았던 날짜를 저장할 변수
f =open('seoul.csv')
data = csv.reader(f)
header =next(data)
for row in data :
    if row[-1] =='' :
        row[-1] =-999   # -999를 넣어 빈 문자열이 있던 자리라고 표시
    row[-1] = float(row[-1])
    if max_temp < row[-1] :
        max_date = row[0]
        max_temp = row[-1]
f.close()
print('기상 관측 이래 서울의 최고 기온이 가장 높았던 날은',max_date+'로, ', max_temp, '도 였습니다.')

==
import matplotlib.pyplot as plt
==
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

plt.plot(result, 'r') # result 리스트에 저장된 값을 빨간색 그래프로 그리기
plt.show() # 그래프 나타내기
=======
```
# 내장함수
- abs(X):<br>
숫자 x의 절대값 반환
- all(iterable_x):<br>
그룹 자료형 변수 x의 모든 원소가 참(0이 아닌 값)이면 True 반환
- any(iterable_x):<br>
그룹 자료형 변수 x의 원소 중 하나라도 참(0이 아닌 값)이면 True 반환
- chr(x):<br>
아스키 코드 값 x에 대한문자 출력
- ord(c):<br>
문자 c에 대한 아스키코드 값 출력
- divmod(a,b):<br>
a를 b로 나눈 몫과 나머지를 튜플로 반환
- int(x):<br>
x를 정수 형태로 반환
- str(x):<br>
x를 문자열 형태로 반환
- list(x):<br>
x를 리스트 형태로 반환
- tuple(x):<br>
x를 투플 형태로 반환
- type(x):<br>
x의 자료형 반환
- lambda :<br>
간단한 삽입형 함수 생성
- max(iterable_x):<br>
반복 가능한 그룹 자료형 x를 입력 받은 뒤 최대값 반환
- min(iterable_x):<br>
반복 가능한 그룹 자료형 x를 입력 받은 뒤 최소값 반환
- pow(x,y):<br>
x의 y제곱 결과값 반환
- input():<br>
사용자 입력으로 받은 값을 문자열로 반환
- range(x):<br>
입력 받은 숫자에 해당되는 범위의 값을 반환
- len(s):<br>
입력값 s의 길이를 반환
- sorted(iterable_x):<br>
입력값을 정렬하여 리스트로 반환

# numpy
- 수치 데이터를 다루기 위한 라이브러리
- 다차원 배열 자료구조인 ndarray를 지원
- 선형대수 계산 등의 행렬 연산에 주로 사용
- 파이썬만 설치한 경우 기본 라이브러리가 아니므로 설치 필요
- 아나콘다 설치시 자동으로 설치됨
- 임포트 및 버전 확인
- np.array():<br>
리스트를 사용한 배열 생성
- np.arange(시작,끝값,간격):<br>
시작, 끝 인덱스, 간격기준으로 출력
- np.array().reshape():<br>
구조를 지정하여 배열 생성
- np.zeros():<br>
초기값과 구조를 지정하여 배열 생성
- 슬라이싱<br>
ar6 = ar2[0:2, 0:2]
- 사칙연산<br>
ar8 = ar1+10 # ar1=[1,2,3,4,5]
ar8 = [11,12,13,14,15]
- np.dot():<br>
행렬 곱연산 ex) ar2 = np.dot(ar1,ar3)
ar1의 열과 ar3의 행의 차수가 같아야한다.
- 헤더<br>
표지판 역할, 데이터 파일에서 여러 가지 값들이 어떤 의미를 갖는지 표시한 행<br>
일반적으로 첫행 위치
- next<br>
한 행씩 읽는 함수<br>
현재 위치의 데이터 행을 읽어오면서 탐색 위치를 다음 행으로 이동시키는 명령
# 기온 데이터 분석
- 데이터 분석은 관심있는 데이터에 대한 호기심에서 출발
- 기온데이터에 대한 호기심<br>
서울이 가장 더웠던 날은? 얼마나 더웠을까?<br>
일교차가 가장 큰 시기는 1년 중 언제쯤일까?<br>
겨울에는 언제가 가장 추울까??

- 질문
서울이 가장 더웠던 날은 언제이고 얼마나 더웠을까?
- 기온, 습도, 풍속 영향이 있겠지만 기온만 대상으로 분석
- 질문을 "서울의 최고 기온은 언제이고 몇 도일까?"로 수정
- seoul.csv에서 필요 필드는 '날짜'와 '최고기온'
- 알고리즘<br>
1. 데이터 읽기
2. 순차적으로 최고 기온 확인
3. 최고 기온이 가장 높았던 날자의 데이터 저장
4. 최종 저장된 데이터 출력
- 빈 문자열을 적당한 값 -999로 채우기
row[-1]=-999 # -999를 넣어 빈 문자열이 있던 자리라고 표시
```python
- 서울의 최고온도 그래프 그리기
import matplotlib.pyplot as plt
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []
for row in data :
    if row[-1] ! = '':
        result.append(float(row[-1]))
plt.plot(result,'r')
plt.show()
# 사용 방법
import numpy as np
```