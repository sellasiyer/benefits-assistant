package com.sellasiyer.benefits.gateway;

/** Thrown when the Python AI service can't be reached after retries, or its circuit is open. */
public class AiServiceUnavailableException extends RuntimeException {

    public AiServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
