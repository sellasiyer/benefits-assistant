package com.sellasiyer.benefits.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * Spring Boot API gateway for the Benefits Assistant.
 *
 * <pre>
 *   React UI  ──►  this gateway (Java, :8080)  ──►  Python AI service (FastAPI + LangGraph, :8000)
 * </pre>
 *
 * Java owns the public API contract, validation, resilience and auditing; the Python service owns
 * the GenAI work (retrieval, the agent, the LLM calls) and scales independently.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
