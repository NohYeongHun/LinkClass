# JPA
## 개념
- 하이버네이트 개발자들이 중심이 되어 만든 ORM 표준이 바로 JPA(Java Perstience API)이다.
- JPA가 제공하는 인터페이스를 이용하여 데이터베이스를 처리하면 실제로는 JPA를 구현한 구현체가 동작한다.
- 스프링 부트에서는 하이버네이트를 JPA 구현체로 이용한다.

## JPA 동작 원리
- JPA는 자바 객체를 컬렉션에 저장하고 관리하는 것과 비슷한 개념이다.
- JPA는 자바 애플리케이션과 JDBC 사이에 존재하면서 JDBC의 복잡한 절차를 대신 처리해준다.
- JPA가 데이터베이스 연동에 사용되는 코드뿐만 아니라 SQL까지 제공. => 메소드 사용시 QUERY문이 날라간다.
- JPA를 이용해서 데이터베이스 연동을 처리하면 개발 및 유지보수의 편의성이 극대화된다.
- 테이블과 VO 클래스 이름을 똑같이 매핑하고, 테이블의 컬럼을 VO 클래스의 멤버 변수에 매핑하면 얼마든지 VO를 기준으로 어느정도 획일화된 SQL을 생성 가능하다. => 예시로 DTO 클래스가 있다.

## JPA 장점
- 개발이 편리하다. => CRUD부분에서 많은 부분이 자동화된다.
- 데이터베이스에 독립적인 개발이 가능하다. 
- 유지보수가 쉽다.

## JPA의 단점
- 학습곡선이 크다. 즉 배워야 할 것이 많다.
- 특정 데이터베이스에 기능을 사용할 수 없다.
- 객체지향 설계가 필요하다.

## RestFull 서비스
- GET(read), POST(create), PUT(update), DELETE(delete)=> RESTFULL 서비스

## reference
[쿼리메소드]https://happygrammer.tistory.com/158