package com.sellasiyer.benefits.springai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * The Benefits Assistant RAG agent implemented natively in Java with Spring AI.
 * Same API contract as the Python service, so the React UI can point at either.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class SpringAiApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAiApplication.class, args);
    }
}
