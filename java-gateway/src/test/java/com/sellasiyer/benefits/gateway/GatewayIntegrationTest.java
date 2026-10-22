package com.sellasiyer.benefits.gateway;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;

/**
 * Boots the real gateway on a random port against a stub "Python AI service" (a JDK HttpServer),
 * so the tests exercise HTTP, JSON mapping, validation, retries, the circuit breaker and SSE relay
 * without needing Python running.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class GatewayIntegrationTest {

    static HttpServer stub;
    static final AtomicInteger chatCalls = new AtomicInteger();
    static final AtomicInteger failuresRemaining = new AtomicInteger();
    static final AtomicReference<String> lastRequestId = new AtomicReference<>();

    static final String CHAT_JSON = """
            {"request_id":"py1","answer":"A specialist office visit has a $50 copay. [1]","route":"generate",
             "citations":[{"id":1,"source":"plan_overview.md","section":"Copays","score":0.31,"text":"..."}],
             "redacted":[],"trace":["guard: clean","router: rag"],"latency_ms":7}""";

    static final String SSE_BODY = """
            event: step
            data: {"node": "guard", "trace": ["guard: clean"]}

            event: token
            data: {"text": "A specialist "}

            event: token
            data: {"text": "visit has a $50 copay. [1]"}

            event: citations
            data: []

            event: done
            data: {"route": "generate", "redacted": []}

            """;

    @LocalServerPort
    int port;

    @Autowired
    CircuitBreakerRegistry circuitBreakers;

    final HttpClient http = HttpClient.newHttpClient();
    final ObjectMapper json = new ObjectMapper();

    @BeforeAll
    static void startStub() throws IOException {
        stub = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        stub.createContext("/health", ex -> send(ex, 200, "application/json",
                "{\"status\":\"ok\",\"llm\":\"offline\",\"embeddings\":\"offline\",\"chunks\":19}"));
        stub.createContext("/chat/stream", ex -> send(ex, 200, "text/event-stream", SSE_BODY));
        stub.createContext("/chat", ex -> {
            chatCalls.incrementAndGet();
            lastRequestId.set(ex.getRequestHeaders().getFirst("X-Request-Id"));
            ex.getRequestBody().readAllBytes();
            if (failuresRemaining.getAndUpdate(n -> Math.max(0, n - 1)) > 0) {
                send(ex, 500, "application/json", "{\"detail\":\"boom\"}");
            } else {
                send(ex, 200, "application/json", CHAT_JSON);
            }
        });
        stub.createContext("/documents", ex -> {
            ex.getRequestBody().readAllBytes();
            send(ex, 200, "application/json", "{\"source\":\"new.md\",\"chunks\":3}");
        });
        stub.start();
    }

    @AfterAll
    static void stopStub() {
        stub.stop(0);
    }

    @DynamicPropertySource
    static void aiServiceUrl(DynamicPropertyRegistry registry) {
        registry.add("ai-service.base-url", () -> "http://127.0.0.1:" + stub.getAddress().getPort());
    }

    @BeforeEach
    void reset() {
        chatCalls.set(0);
        failuresRemaining.set(0);
        lastRequestId.set(null);
        circuitBreakers.circuitBreaker("aiService").reset();
    }

    @Test
    void healthAggregatesAiServiceAndNamesTheBackend() throws Exception {
        JsonNode body = json.readTree(get("/health").body());
        assertThat(body.get("status").asText()).isEqualTo("ok");
        assertThat(body.get("chunks").asInt()).isEqualTo(19);
        assertThat(body.get("backend").asText()).contains("spring-boot-gateway");
    }

    @Test
    void chatIsProxiedWithTheSameContract() throws Exception {
        HttpResponse<String> resp = postJson("/chat", "{\"question\":\"What is the specialist copay?\"}", null);
        assertThat(resp.statusCode()).isEqualTo(200);
        JsonNode body = json.readTree(resp.body());
        assertThat(body.get("answer").asText()).contains("$50");
        assertThat(body.get("citations").get(0).get("source").asText()).isEqualTo("plan_overview.md");
        assertThat(body.has("request_id")).isTrue();
    }

    @Test
    void requestIdIsPropagatedToTheAiServiceAndEchoed() throws Exception {
        HttpResponse<String> resp = postJson("/chat", "{\"question\":\"What is the specialist copay?\"}", "trace-42");
        assertThat(lastRequestId.get()).isEqualTo("trace-42");
        assertThat(resp.headers().firstValue("X-Request-Id")).contains("trace-42");
    }

    @Test
    void invalidQuestionIsRejectedBeforeCallingTheAiService() throws Exception {
        HttpResponse<String> resp = postJson("/chat", "{\"question\":\"\"}", null);
        assertThat(resp.statusCode()).isEqualTo(400);
        assertThat(json.readTree(resp.body()).get("error").asText()).isEqualTo("validation_failed");
        assertThat(chatCalls.get()).isZero();
    }

    @Test
    void transientFailuresAreRetried() throws Exception {
        failuresRemaining.set(2); // two 500s, then success
        HttpResponse<String> resp = postJson("/chat", "{\"question\":\"What is the specialist copay?\"}", null);
        assertThat(resp.statusCode()).isEqualTo(200);
        assertThat(chatCalls.get()).isEqualTo(3);
    }

    @Test
    void persistentFailureReturns503WithAClearError() throws Exception {
        failuresRemaining.set(100);
        HttpResponse<String> resp = postJson("/chat", "{\"question\":\"What is the specialist copay?\"}", null);
        assertThat(resp.statusCode()).isEqualTo(503);
        assertThat(json.readTree(resp.body()).get("error").asText()).isEqualTo("ai_service_unavailable");
        assertThat(chatCalls.get()).isEqualTo(3); // max-attempts
    }

    @Test
    void streamingRelaysEverySseFrameInOrder() throws Exception {
        HttpResponse<String> resp = postJson("/chat/stream", "{\"question\":\"What is the specialist copay?\"}", null);
        assertThat(resp.statusCode()).isEqualTo(200);
        String body = resp.body();
        assertThat(body).contains("event:step", "event:token", "event:citations", "event:done");
        assertThat(body.indexOf("event:step")).isLessThan(body.indexOf("event:done"));
        assertThat(body).contains("visit has a $50 copay. [1]");
    }

    // ---------------------------------------------------------------- helpers
    private HttpResponse<String> get(String path) throws Exception {
        return http.send(HttpRequest.newBuilder(uri(path)).GET().build(), HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> postJson(String path, String body, String requestId) throws Exception {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri(path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body));
        if (requestId != null) b.header("X-Request-Id", requestId);
        return http.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    private URI uri(String path) {
        return URI.create("http://127.0.0.1:" + port + path);
    }

    private static void send(HttpExchange ex, int status, String type, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", type);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }
}
