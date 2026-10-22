package com.sellasiyer.benefits.springai;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.ClassPathResource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sellasiyer.benefits.springai.core.CostMath;

/**
 * Model and vector-store wiring.
 *
 * <p>Provider choice is configuration: with {@code app.llm-provider=offline} (the default) the offline
 * models are used; run with a Maven profile + Spring profile (e.g. {@code -Popenai}) and the provider's
 * Spring AI starter auto-configures a real {@link ChatModel} / {@link EmbeddingModel} instead. Nothing
 * else in the app changes, because everything depends on the Spring AI interfaces.
 *
 * <p>{@link SimpleVectorStore} is in-memory; swap the bean for PgVectorStore, AzureVectorStore,
 * OpenSearchVectorStore, etc. in production.
 */
@Configuration
public class AiConfig {

    @Bean
    @Primary
    @ConditionalOnProperty(name = "app.llm-provider", havingValue = "offline", matchIfMissing = true)
    ChatModel offlineChatModel() {
        return new OfflineChatModel();
    }

    @Bean
    @Primary
    @ConditionalOnProperty(name = "app.embedding-provider", havingValue = "offline", matchIfMissing = true)
    EmbeddingModel offlineEmbeddingModel() {
        return new HashingEmbeddingModel();
    }

    @Bean
    ChatClient chatClient(ChatModel chatModel) {
        return ChatClient.create(chatModel);
    }

    @Bean
    VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }

    @Bean
    CostMath costMath(ObjectMapper json) throws IOException {
        try (InputStream in = new ClassPathResource("cost_table.json").getInputStream()) {
            JsonNode t = json.readTree(in);
            Map<String, Double> copays = new HashMap<>();
            t.get("copays").fields().forEachRemaining(e -> copays.put(e.getKey(), e.getValue().asDouble()));
            return new CostMath(copays, t.get("coinsurance_in_network").asDouble(), t.get("oop_max_individual").asDouble());
        }
    }
}
