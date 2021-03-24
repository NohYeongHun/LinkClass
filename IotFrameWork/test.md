eclipse -> help -> marketplace -> spring tools 3 (Standalone Edition) 3.9.1.4
sts
Package Explorer -> new
Start Starter Project
Start Legacy Project
Spring MVC Project
=> kr.inhatc.spring
Eclipse IDE for Enterprise Java Developers

alt+shift+s => getter setter and 생성자 생성

# Eclipse Default Encoding
- Window -> Preference -> General -> Content Type -> Default Encoding = UTF-8 Update

# Sprint Project
1. New-> SpringStartProject -> Type Maven
2. Next 
- Developer Tools -> Spring Boot DevTools
- Spring Web

3. porm.xml
- 라이브러리 모음, 버전 관리

4. Maven Update
- 프로젝트 설정 후 -> 오른쪽 클릭 -> 메이븐 -> 메이븐 업데이트

5. application.properties
- server.port=18080

6. Running
- Run as -> spring boot app
- chrome -> localhost:18080

7. 실행순서
- html 찾으러 먼저옴 

8. porm.xml
-> depency에서 ctrl+space

9. open with
-> others ->visual studio code 선택 가능

# 추가 해야하는 툴
- Spring dev tools , Mybatis FrameWork, 

# DBeaver oracle connect
- Oracle 11g enterprise
- DBeaver 21.0 community
## Run SQL Command Line 
```SQL
SQL> conn / as sysdba
Connected.
SQL> show user
USER is "SYS"
SQL> alter user hr identified by spring account unlock;

User altered.

SQL> grant connect, resource to hr;

Grant succeeded.

SQL> conn hr/spring
Connected.
SQL> show user
USER is "HR"
SQL> conn / as sysdba
Connected.
SQL> show user
USER is "SYS"

```
# 소스 코드

## application.properties
```yml
# DB 설정 (hikari CP란?)
spring.datasource.hikari.driver-class-name=oracle.jdbc.driver.OracleDriver
spring.datasource.hikari.jdbc-url=jdbc:oracle:thin:@127.0.0.1:1521:xe
spring.datasource.hikari.username=hr
spring.datasource.hikari.password=spring
spring.datasource.hikari.connection-timeout=10000
spring.datasource.hikari.validation-timeout=10000
spring.datasource.hikari.connection-test-query=SELECT 1 FROM DUAL

# 포트 설정
server.port=18080
```

## DataBaseConfiguration.java
```java
package kr.inhatc.spring.configuration;

import javax.sql.DataSource;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

@Configuration
@PropertySource("classpath:/application.properties")
public class DataBaseConfiguration {
	@Bean
	@ConfigurationProperties(prefix="spring.datasource.hikari")
	public HikariConfig hikariConfig() {
		return new HikariConfig();
	}
	
	@Bean
	public DataSource dataSource() throws Exception{
		DataSource dataSource = new HikariDataSource(hikariConfig());
		System.out.println("================="+dataSource.toString());
		return dataSource;
	}
	
}

```
## BoardController.java
```java
package kr.inhatc.spring.board.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class BoardController {
	@RequestMapping("/")
	public String hello() {
		return "index";
	}
}

```
## porm.xml
```xml

<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	<modelVersion>4.0.0</modelVersion>
	<parent>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-starter-parent</artifactId>
		<version>2.4.3</version>
		<relativePath/> <!-- lookup parent from repository -->
	</parent>
	<groupId>kr.inhatc.spring</groupId>
	<artifactId>MyProject</artifactId>
	<version>0.0.1-SNAPSHOT</version>
	<name>MyProject</name>
	<description>Basic project for Spring Boot</description>
	<properties>
		<java.version>15</java.version>
	</properties>
	<dependencies>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-data-jpa</artifactId>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-thymeleaf</artifactId>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-web</artifactId>
		</dependency>
		<dependency>
			<groupId>org.mybatis.spring.boot</groupId>
			<artifactId>mybatis-spring-boot-starter</artifactId>
			<version>2.1.4</version>
		</dependency>

		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-devtools</artifactId>
			<scope>runtime</scope>
			<optional>true</optional>
		</dependency>
		<dependency>
			<groupId>com.oracle.database.jdbc</groupId>
			<artifactId>ojdbc8</artifactId>
			<scope>runtime</scope>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-test</artifactId>
			<scope>test</scope>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-configuration-processor</artifactId>
			<optional>true</optional>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-configuration-processor</artifactId>
			<optional>true</optional>
		</dependency>
	</dependencies>

	<build>
		<plugins>
			<plugin>
				<groupId>org.springframework.boot</groupId>
				<artifactId>spring-boot-maven-plugin</artifactId>
			</plugin>
		</plugins>
	</build>

</project>

```
