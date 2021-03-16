# 내적
1. 내적은 스칼라 값이다.
- x.inner_product(y)
a(inner)
|a-b|^2 = (a-b)

2. 코사인 제 2 법칙
- cosΘ = a·b/|a||b|

3. 코시-슈바르츠 부등식
- |a·b|<=|a||b|
(단, 등호는 a,b 중 하나가 다른 것의 실수배일 때만 성립)

a·b = |a||b|cosΘ
Θ = 0 a//b(같은 방향으로 평행) a·b = |a||b|
Θ = π/2 a(직교)b = a·b=0 
Θ = π a//b ( 반대 방향으로 평행) a·b = -|a||b|

- 그림자 길이의 공식 : comp(a)b = |b|cosΘ = |a||b|cosΘ/|a| =a·b/|a|

4. vector projection of b onto a
- a에 수직인 b의 성분을 w = SR->라 하면 
w = b - proj(a)b이다.
proj(a)b = comp(a)b*a/|a| = (a·b/|a|^2)*a

# 외적
- x.cross_prdouct(y)
|a b|
|c d| = > ad-bc
a=(a1,a2,a3), b=(b1,b2,b3)
a*b = (|a2 a3|b2 b3|,|a1 a3|b1 b3|,|a1 a2|b1 b2|)
= (a2b3 -a3b2 -(a1b3-a3b1), a1b2-a2b1)

외적 a*b는 a와 b에 각각 수직이다.

벡터 a*b의 길이 |a*b|=|a||b|sinΘ
|a*b|^2 = |a|^2|b|^2 - (a-b)^2

# 평행한 경우
