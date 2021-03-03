bonus = 0
price = int(input( "단가 입력" ))
amt = int(input( "수량 입력" ))
while price != 0:
  result = price * amt
  print("result 총액 : ", result)
  if result >= 500:
    bonus += result * 0.1
  else if result>=300:
    bonus += result * 0.01
  print("bonuse 총액:", bonus)
  price = int(input( "단가 입력" ))
  amt = int(input( "수량 입력" ))
