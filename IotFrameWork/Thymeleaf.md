# Thymeleaf 템플릿 캐시 설정

- 스프링부트의 Thymeleaf 테믚릿 결과는 캐싱하는 것이 디폴트 값.
- 즉, 개발 시에 Thymeleaf를 수정하고 브라우저를 새로고침하면 바로 반영이 되지 않는다.
- 따라서 개발을 할 때에는 false로 해 주는 것이 재시작 없이 새로고침만으로 반영되게 하는 것이 편하다.

- application.yml
```yml
spring:
    thymeleaf:
    cache: false
```

- application.properties
```java
spring.thymeleaf.cache=false
```