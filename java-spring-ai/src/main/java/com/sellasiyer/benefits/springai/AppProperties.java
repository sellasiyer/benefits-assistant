package com.sellasiyer.benefits.springai;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * RAG tuning knobs and provider selection, bound from {@code app.*}.
 *
 * @param llmProvider       offline | openai | azure | anthropic
 * @param embeddingProvider offline | openai | azure
 * @param topK              chunks to retrieve
 * @param minRelevance      below this cosine similarity a chunk is dropped; tune per embedding model
 * @param chunkSize         max characters per chunk
 */
@ConfigurationProperties(prefix = "app")
public record AppProperties(String llmProvider, String embeddingProvider, int topK, double minRelevance, int chunkSize) {

    public AppProperties {
        if (llmProvider == null) llmProvider = "offline";
        if (embeddingProvider == null) embeddingProvider = "offline";
        if (topK <= 0) topK = 4;
        if (minRelevance <= 0) minRelevance = 0.15;
        if (chunkSize <= 0) chunkSize = 400;
    }

    public boolean offlineLlm() {
        return "offline".equalsIgnoreCase(llmProvider);
    }
}
