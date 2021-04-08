# 스트링
```Matlab
>> s1 = ['a' 'b' 'c' 'd' 'e']

s1 =

    'abcde'

>> whos s1
  Name      Size            Bytes  Class    Attributes

  s1        1x5                10  char               

>> s1(1)

ans =
  
      'a'
>> s2 = s1'

s2 =
      
    5×1 char 배열
      
          'a'
          'b'
          'c'
          'd'
          'e'  
>> s1( 2 : 4)

          ans =
          
              'bcd'

              >> s1(1, 1:5)

              ans =
              
                  'abcde'
              
              >> whos
                Name        Size            Bytes  Class     Attributes
              
                A           3x1                24  double              
                C         100x80            64000  double              
                a           3x1                24  double              
                ans         1x5                10  char                
                b           1x3                24  double              
                c         100x80            64000  double              
                s1          1x5                10  char                
                s2          5x1                10  char                
              
              >> s_mat =['name1', 'name2','name3']
              
              s_mat =
              
                  'name1name2name3'
              
              >> s_mat(1)
              
              ans =
              
                  'n'
              
              >> whos s_mat
                Name       Size            Bytes  Class    Attributes
              
                s_mat      1x15               30  char               
              
              >> s_mat = [ 'name1' ; 'name2'; 'name3']
              
              s_mat =
              
                3×5 char 배열
              
                  'name1'
                  'name2'
                  'name3'
              
              >> s_mat(1, :)
              
              ans =
              
                  'name1'
              
              >> s_mat(2, :)
              
              ans =
              
                  'name2'
              
              >> s_mat(:,1)
              
              ans =
              
                3×1 char 배열
              
                  'n'
                  'n'
                  'n'
              
              >> s_mat(:,5)
              
              ans =
              
                3×1 char 배열
              
                  '1'
                  '2'
                  '3'
 ```         
# 변수 및 벡터
```Matlab
>> a = [10 20 30 40]

a =

    10    20    30    40

>> b = [2 2 2 2]

b =

     2     2     2     2

>> a.*b

ans =

    20    40    60    80

>> a = [1 : 10 ]

a =

     1     2     3     4     5     6     7     8     9    10

>> a

a =

     1     2     3     4     5     6     7     8     9    10

>> a = [ 1 : 10 ]

a =

     1     2     3     4     5     6     7     8     9    10

>> a=[1: 5: 10]

a =

     1     6

>> a=[1 : 6 : 11]

a =

     1     7

>> a=[1 : 5 : 11]

a =

     1     6    11

>> whos
  Name      Size            Bytes  Class     Attributes

  a         1x3                24  double              
  ans       1x4                32  double              
  b         1x4                32  double              

>> 
>> a=[1;2;3]

a =

     1
     2
     3

>> whos
  Name      Size            Bytes  Class     Attributes

  a         3x1                24  double              
  ans       1x4                32  double              
  b         1x4                32  double              

>> A = ones(2,2)

A =

     1     1
     1     1
>> zeros(3)

     ans =
     
          0     0     0
          0     0     0
          0     0     0
>> eye(2)
       ans =
          
       1     0
       0     1

  >> eye(3)
          
          ans =
          
               1     0     0
               0     1     0
               0     0     1
          
          >> eye(2,3)
          
          ans =
          
               1     0     0
               0     1     0
               A =

               1     1
               1     1
               1     1
          
          >> A = ones (1, 3)
          
          A =
          
               1     1     1
          
          >> A = ones (3,1)
          
          A =
          
               1
               1
               1
```
# 셀만들기
```Matlab
>> A = [1 2 ; 3 4]

A =

     1     2
     3     4

>> c = {1}

c =

  1×1 cell 배열

    {[1]}

>> whos c
  Name      Size            Bytes  Class    Attributes

  c         1x1               112  cell               

>> c = {'abcde'}

c =

  1×1 cell 배열

    {'abcde'}

>> whos c
  Name      Size            Bytes  Class    Attributes

  c         1x1               114  cell               

>> c{1} =1

c =

  1×1 cell 배열

    {[1]}

>> c{2} ='abcde'

c =

  1×2 cell 배열

    {[1]}    {'abcde'}

>> whos c
  Name      Size            Bytes  Class    Attributes

  c         1x2               226  cell               

>> c{3} = {1 2 ; 3 4}

c =

  1×3 cell 배열

    {[1]}    {'abcde'}    {2×2 cell}

>> c{3}

ans =

  2×2 cell 배열

    {[1]}    {[2]}
    {[3]}    {[4]}

>> d {1,1} =1

d =

  1×1 cell 배열

    {[1]}

>> d{1,2} ='abc'

d =

  1×2 cell 배열

    {[1]}    {'abc'}

>> d{1,3} = [1,2,3]

d =

  1×3 cell 배열

    {[1]}    {'abc'}    {1×3 double}

>> d{1,3}

ans =

     1     2     3

>> d{2,1} = 'abcde'

d =

  2×3 cell 배열

    {[    1]}    {'abc'     }    {1×3 double}
    {'abcde'}    {0×0 double}    {0×0 double}

>> whos d
  Name      Size            Bytes  Class    Attributes

  d         2x3               480  cell               

>> d{2,2} = [12; 34]

d =

  2×3 cell 배열

    {[    1]}    {'abc'     }    {1×3 double}
    {'abcde'}    {2×1 double}    {0×0 double}

>> d{2,2} = [1 2 ; 3 4]

d =

  2×3 cell 배열

    {[    1]}    {'abc'     }    {1×3 double}
    {'abcde'}    {2×2 double}    {0×0 double}

>> d{2,3} = [5; 5; 5]

d =

  2×3 cell 배열

    {[    1]}    {'abc'     }    {1×3 double}
    {'abcde'}    {2×2 double}    {3×1 double}

>> whos s1 s2 s3
  Name      Size            Bytes  Class    Attributes

  s1        1x5                10  char               
  s2        5x1                10  char               

>> whos s1 s2 d
  Name      Size            Bytes  Class    Attributes

  d         2x3               728  cell               
  s1        1x5                10  char               
  s2        5x1                10  char               

>> s_list ={ s1,s2}

s_list =

  1×2 cell 배열

    {'abcde'}    {5×1 char}

>> s1 = youngsoo
'youngsoo'은(는) 인식할 수 없는 함수 또는 변수입니다.
 
>> s1 ='youngsoo
 s1 ='youngsoo
     ↑
오류: 문자형 벡터가 제대로 종료되지 않았습니다.
 
>> s1 ='youngsoo'

s1 =

    'youngsoo'

>> s2 = 'chulsoo'

s2 =

    'chulsoo'

>> s3 ='boram'

s3 =

    'boram'

>> s_list={s1,s2,s3}

s_list =

  1×3 cell 배열

    {'youngsoo'}    {'chulsoo'}    {'boram'}

>> whos s_list
  Name        Size            Bytes  Class    Attributes

  s_list      1x3               352  cell               

>> s_list={s1 ; s2; s2}

s_list =

  3×1 cell 배열

    {'youngsoo'}
    {'chulsoo' }
    {'chulsoo' }

>> whos s_list
  Name        Size            Bytes  Class    Attributes

  s_list      3x1               356  cell               

>> whost s_list2
'whost'은(는) 인식할 수 없는 함수 또는 변수입니다.
 
정정 제안:
>> s_list2={s1 ; s2; s2}

s_list2 =

  3×1 cell 배열

    {'youngsoo'}
    {'chulsoo' }
    {'chulsoo' }

>>  s_list={s1,s2,s3}

s_list =

  1×3 cell 배열

    {'youngsoo'}    {'chulsoo'}    {'boram'}

>> whos s_list2
  Name         Size            Bytes  Class    Attributes

  s_list2      3x1               356  cell               

>> s_list3 = {1, [2 3 ], 'abc' ; 'de', [1 2 ; 3 4], 'e'}

s_list3 =

  2×3 cell 배열

    {[ 1]}    {1×2 double}    {'abc'}
    {'de'}    {2×2 double}    {'e'  }

>> s_list4{1,1} = 1

s_list4 =

  1×1 cell 배열

    {[1]}
```

# 스트럭
```Matlab
>> car(1).company = 'company a'

car = 

  다음 필드를 포함한 struct:

    company: 'company a'

>> car(1).color = 'white'

car = 

  다음 필드를 포함한 struct:

    company: 'company a'
      color: 'white'

>> car(1).year = 2019

car = 

  다음 필드를 포함한 struct:

    company: 'company a'
      color: 'white'
       year: 2019

>> car(1).type
존재하지 않는 필드 'type'에 대한 참조입니다.
 
>> car(1).type ='sedan'

car = 

  다음 필드를 포함한 struct:

    company: 'company a'
      color: 'white'
       year: 2019
       type: 'sedan'

>> whos car
  Name      Size            Bytes  Class     Attributes

  car       1x1               718  struct              

>> car(2).company = 'company b'

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type

>> car

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type

>> car2.color = 'red'

car2 = 

  다음 필드를 포함한 struct:

    color: 'red'

>> car(2).color ='red'

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type

>> car(2).year='200'

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type

>> car(2).type='suv'

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type
>> car(1)

ans = 

  다음 필드를 포함한 struct:

    company: 'company a'
      color: 'white'
       year: 2019
       type: 'sedan'

>> car(2)

ans = 

  다음 필드를 포함한 struct:

    company: 'company b'
      color: 'red'
       year: '200'
       type: 'suv'

>> car(2).year='2000'

car = 

  다음 필드를 포함한 1×2 struct 배열:

    company
    color
    year
    type

>> car(:).company

ans =

    'company a'


ans =

    'company b'


>> list_company = car(:).company

list_company =

    'company a'

>> list_company

list_company =

    'company a'

>> list_company ={ car(:).company}

list_company =

  1×2 cell 배열

    {'company a'}    {'company b'}

>> list_year = car( :).year

list_year =

        2019

>> list_year={ car(:).year}

list_year =

  1×2 cell 배열

    {[2019]}    {'2000'}

```