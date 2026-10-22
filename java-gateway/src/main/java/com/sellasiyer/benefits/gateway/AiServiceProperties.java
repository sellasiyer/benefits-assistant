package com.sellasiyer.benefits.gateway;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** Where the Python AI service lives and how long we wait for it. Bound from {@code ai-service.*}. */
@ConfigurationProperties(prefix = "ai-service")
public record AiServiceProperties(String baseUrl, Duration connectTimeout, Duration readTimeout) {

    public AiServiceProperties {
        if (baseUrl == null || baseUrl.isBlank()) baseUrl = "http://localhost:8000";
        if (connectTimeout == null) connectTimeout = Duration.ofSeconds(2);
        if (readTimeout == null) readTimeout = Duration.ofSeconds(30);
    }
}
