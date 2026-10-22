package com.sellasiyer.benefits.gateway;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sellasiyer.benefits.gateway.Dtos.ChatRequest;
import com.sellasiyer.benefits.gateway.Dtos.ChatResponse;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;

/**
 * Typed client for the Python AI service.
 *
 * <ul>
 *   <li>Request/response calls use Spring's {@link RestClient} with explicit connect/read timeouts.</li>
 *   <li>{@code @Retry} retries transient failures (connection refused, 5xx) with backoff; 4xx are never retried.</li>
 *   <li>{@code @CircuitBreaker} stops hammering a failing AI service and fails fast with a clear 503.</li>
 *   <li>Streaming uses the JDK {@link HttpClient} so we can relay the SSE body line by line.</li>
 * </ul>
 */
@Component
public class AiServiceClient {

    private final RestClient rest;
    private final HttpClient http;
    private final AiServiceProperties props;
    private final ObjectMapper json;

    public AiServiceClient(RestClient.Builder builder, AiServiceProperties props, ObjectMapper json) {
        this.props = props;
        this.json = json;
        this.http = HttpClient.newBuilder().connectTimeout(props.connectTimeout()).build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(http);
        factory.setReadTimeout(props.readTimeout());
        this.rest = builder.baseUrl(props.baseUrl()).requestFactory(factory).build();
    }

    @Retry(name = "aiService")
    @CircuitBreaker(name = "aiService", fallbackMethod = "chatFallback")
    public ChatResponse chat(ChatRequest request, String requestId) {
        return rest.post().uri("/chat")
                .header(RequestIdFilter.HEADER, requestId)
                .contentType(MediaType.APPLICATION_JSON)
                .body(request)
                .retrieve()
                .body(ChatResponse.class);
    }

    // invoked reflectively by resilience4j when a call fails or the circuit is open
    ChatResponse chatFallback(ChatRequest request, String requestId, Throwable t) {
        if (t instanceof HttpClientErrorException clientError) throw clientError; // caller's fault: pass through
        throw new AiServiceUnavailableException("AI service unavailable: " + t.getMessage(), t);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> health() {
        return rest.get().uri("/health").retrieve().body(Map.class);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> upload(String filename, byte[] content) {
        var parts = new LinkedMultiValueMap<String, Object>();
        parts.add("file", new ByteArrayResource(content) {
            @Override
            public String getFilename() {
                return filename;
            }
        });
        return rest.post().uri("/documents")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(Map.class);
    }

    /**
     * Calls the Python streaming endpoint and hands each complete SSE frame (event, data) to {@code onFrame}.
     * Blocks until the stream ends, so call it off the request thread.
     */
    public void stream(ChatRequest request, String requestId, BiConsumer<String, String> onFrame)
            throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(props.baseUrl() + "/chat/stream"))
                .timeout(props.readTimeout())
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header(RequestIdFilter.HEADER, requestId)
                .POST(HttpRequest.BodyPublishers.ofString(toJson(request)))
                .build();
        HttpResponse<Stream<String>> resp = http.send(req, HttpResponse.BodyHandlers.ofLines());
        if (resp.statusCode() >= 400) {
            throw new AiServiceUnavailableException("AI service returned HTTP " + resp.statusCode(), null);
        }
        String event = "message";
        StringBuilder data = new StringBuilder();
        try (Stream<String> lines = resp.body()) {
            for (String line : (Iterable<String>) lines::iterator) {
                if (line.isEmpty()) { // blank line = end of one SSE frame
                    if (!data.isEmpty()) onFrame.accept(event, data.toString());
                    event = "message";
                    data.setLength(0);
                } else if (line.startsWith("event:")) {
                    event = line.substring(6).trim();
                } else if (line.startsWith("data:")) {
                    data.append(line.substring(5).trim());
                }
            }
        }
        if (!data.isEmpty()) onFrame.accept(event, data.toString());
    }

    private String toJson(Object value) {
        try {
            return json.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException(e);
        }
    }
}
