package com.sellasiyer.benefits.gateway;

import java.util.List;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * The public API contract. Identical field names to the Python service's Pydantic models,
 * so the React UI works against either backend unchanged.
 */
public final class Dtos {

    private Dtos() {
    }

    public record ChatRequest(
            @NotBlank(message = "question is required")
            @Size(min = 3, max = 2000, message = "question must be 3-2000 characters") String question) {
    }

    public record Citation(int id, String source, String section, double score, String text) {
    }

    public record ChatResponse(
            String request_id,
            String answer,
            String route,
            List<Citation> citations,
            List<String> redacted,
            List<String> trace,
            int latency_ms) {
    }

    public record ErrorResponse(String error, String detail, String request_id) {
    }
}
