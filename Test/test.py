'''
수를 처리하는 것은 통계학에서 상당히 중요한 일이다. 통계학에서 N개의 수를 대표하는 기본 통계값에는 다음과 같은 것들이 있다. 단, N은 홀수라고 가정하자.

1. 산술평균 : N개의 수들의 합을 N으로 나눈 값
2. 중앙값 : N개의 수들을 증가하는 순서로 나열했을 경우 그 중앙에 위치하는 값
3. 최빈값 : N개의 수들 중 가장 많이 나타나는 값
4. 범위 : N개의 수들 중 최댓값과 최솟값의 차이

입력:
첫째 줄에 수의 개수 N(1 ≤ N ≤ 500,000)이 주어진다. 그 다음 N개의 줄에는 정수들이 주어진다. 입력되는 정수의 절댓값은 4,000을 넘지 않는다.

출력
첫째 줄에는 산술평균을 출력한다. 소수점 이하 첫째 자리에서 반올림한 값을 출력한다.
둘째 줄에는 중앙값을 출력한다.
셋째 줄에는 최빈값을 출력한다. 여러 개 있을 때에는 최빈값 중 두 번째로 작은 값을 출력한다.
넷째 줄에는 범위를 출력한다.
'''

# 오름차순 퀵정렬.
def quick_sort(arr):
    if len(arr) <= 1: # 배열의 크기가 1이면 정렬할 인덱스가 없으므로 리턴.
        return arr
    pivot = arr[len(arr) // 2] # 배열의 인덱스가 중앙인 값을 구함.
    lesser_arr, equal_arr, greater_arr = [], [], [] # 적을경우, 동등할경우, 클 경우
    for num in arr: 
        if num < pivot: # pivot 기준 왼쪽 배열
            lesser_arr.append(num)
        elif num > pivot: # pivot 기준 오른쪽 배열 
            greater_arr.append(num) 
        else: # pivot과 동일한 값 배열
            equal_arr.append(num)
    return quick_sort(lesser_arr) + equal_arr + quick_sort(greater_arr)

N = 0
lista =[]
N = int(input('첫 번째 숫자를 입력하세요: '))
for i in range(N):
  s=int(input('리스트에 들어갈 값을 입력하세요 : '))
  lista.append(s)

print("정렬전 리스트 = ",lista)
# 오름차순 정렬
lista = quick_sort(lista)
print("정렬된 리스트 = ",lista)
합계=0
# 1. 산술평균 Arithmetic mean
for i in lista:
    합계 +=i
    Atcmean = 합계/len(lista)
print("산술평균 = ",round(Atcmean))

# 2. 중앙값 : N개의 수들을 증가하는 순서로 나열했을 경우 그 중앙에 위치하는 값
print("중앙값 = ",lista[len(lista)//2])
# 3. 최빈값 : N개의 수들 중 가장 많이 나타나는 값 여러개 있을경우 최빈값중 두번째로 작은값
cnt=0
fincnt=0
choicount=0
result=0
for i in range(N):
    cnt+=1
    if(i==N-1 or lista[i]!=lista[i+1]):
        if(cnt==fincnt):
            fincnt=cnt
            choicount+=1
            if choicount<=2:
                result=lista[i]
        
        elif cnt>fincnt:
            fincnt=cnt
            result=lista[i]
            choicount=0
            choicount+=1
        cnt=0

print("최빈값 = ",result)
# 4. 범위 : N개의 수들 중 최댓값과 최솟값의 차이
print("최댓값-최솟값 = ",max(lista)-min(lista))