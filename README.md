LinkClass
# IOT 서버 프레임워크
1. 책 : 스프링 부트 퀵스타트 
2. JDK : 15버전 이상 EX) JRE System Library [JAVASE - 15]
3. Eclipse 2020-12
4. MarketPlace : Spring
# 산업 AI 임베디드 기초

# AI 모빌리티

# AI 수학 기초
[SageMath-JupyterNoteBook]https://github.com/sagemath/sage-windows/releases

# 머신러닝 기초

# 캡스톤 디자인
1. 주제 선정 :
[뉴스데이터분석참고]https://www.youtube.com/watch?v=Tvi2AgJG0qU
2. 사용 pip : KoreaNewsCrawler 
```python

ex) pip install KoreanNewsCrawler
from korea_news_crawler.articlecrawler import ArticleCrawler
# 기사 뉴스 크롤러의 예시
Crawler = ArticleCrawler()  
Crawler.set_category("정치", "IT과학", "economy")  
Crawler.set_date_range(2017, 1, 2018, 4)  
Crawler.start() 
# 2017년 1월 ~ 2018년 4월까지 정치, IT 과학, 경제 카테고리 뉴스를 멀티보드를 이용하여 송신 크롤링을 진행.
```
3. 방법:
- set_category (카테고리 _ 이름)
이 방법은 수집하려고하는 카테고리는 설정하는 방법입니다.
'정치', '경제', '사회', '생활', 'IT 과학', '세계', '오피니언'입니다.
많은 것은 여러 개 의견 수 있습니다.
category_name : 정치, 경제, 사회, 생활 문화, IT 과학, 세계, 오피니언 또는 정치, 경제, 사회, 생활 _ 문화, IT_ 과학, 세계, 의견

- set_date_range (시작 년, 시작 월, 종료 년, 종료 월)
이 방법은 수집하려고하는 뉴스의 기간을 의미합니다. 기본적으로 startmonth 월부터 endmonth 월까지 데이터를 수집합니다.
[KoreaNewsCrawler]https://github.com/lumyjuwon/KoreaNewsCrawler
# 빅데이터 분석