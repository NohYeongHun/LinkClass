# 이미지
1. 이미지 읽어서 살펴보기
```python
import cv2
cv2.imread(file_name,flag) # 이미지를 읽어 Numpy 객체로 만드는 함수

- file_name : 읽고자 하는 이미지 파일
- flag : 이미지를 읽는 방법 설정
IMREAD_COLOR : 이미지를 Color로 읽고, 투명한 부분은 무시
IMREAD_GRAYSCALE : 이미지를 Grayscale로 읽기
IMREAD_UNCHANGED : 이미지를 Color로 읽고, 투명한 부분도 읽기(Alpha)
반환 값 : Numpy 객체(행, 열, 색상 : 기본 BGR)

cv2.imshow(title, image) # 특정한 이미지를 화면 출력
- title: 윈도우 창의 제목 
- image: 출력할 이미지 객체 
# 구글 colab 사용시 이미지를 출력할 수 없으므로 matplot 라이브러리 사용

cv2.imwrite(file_name, image) # 특정한 이미지를 파일로 저장하는 함수
- file_name : 저장할 이미지 파일 이름
- image : 저장 이미지 객체

cv2.waitKey(time) # 키보드 입력을 처리하는 함수
- tile : 입력 대기 시간 (무한대기 : 0)
# 반환 값 : 사용자가 입력한 ASCII CODE (ESC: 27)

cv2.destoryAllWindows() # 화면의 모든 윈도우를 닫는 함수

```