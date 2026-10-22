package com.sellasiyer.benefits.gateway;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestAttribute;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.sellasiyer.benefits.gateway.Dtos.ChatRequest;
import com.sellasiyer.benefits.gateway.Dtos.ChatResponse;

import jakarta.validation.Valid;

/**
 * Same endpoints and payloads as the Python service (/health, /chat, /chat/stream, /documents),
 * so the UI can point at either. Audit logs record id, route and latency, never the question text,
 * because questions can contain PHI.
 */
@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);
    private static final long STREAM_TIMEOUT_MS = 120_000;

    private final AiServiceClient ai;

    public ChatController(AiServiceClient ai) {
        this.ai = ai;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        try {
            body.putAll(ai.health());
            body.put("backend", "spring-boot-gateway → python-fastapi");
        } catch (Exception e) {
            body.put("status", "degraded");
            body.put("backend", "spring-boot-gateway (AI service unreachable)");
            body.put("llm", "unavailable");
            body.put("embeddings", "unavailable");
            body.put("chunks", 0);
        }
        return body;
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request,
                             @RequestAttribute(RequestIdFilter.ATTR) String requestId) {
        long t0 = System.nanoTime();
        ChatResponse resp = ai.chat(request, requestId);
        log.info("audit rid={} route={} gateway_ms={} ai_ms={}", requestId, resp.route(),
                (System.nanoTime() - t0) / 1_000_000, resp.latency_ms());
        return resp;
    }

    @PostMapping("/chat/stream")
    public SseEmitter stream(@Valid @RequestBody ChatRequest request,
                             @RequestAttribute(RequestIdFilter.ATTR) String requestId) {
        SseEmitter emitter = new SseEmitter(STREAM_TIMEOUT_MS);
        // Relay on a virtual thread: cheap to block on the upstream stream, frees the servlet thread.
        Thread.startVirtualThread(() -> {
            try {
                ai.stream(request, requestId, (event, data) -> {
                    try {
                        emitter.send(SseEmitter.event().name(event).data(data));
                    } catch (IOException e) {
                        throw new IllegalStateException("client disconnected", e);
                    }
                });
                emitter.complete();
                log.info("audit rid={} stream=complete", requestId);
            } catch (Exception e) {
                log.warn("audit rid={} stream=failed reason={}", requestId, e.getMessage());
                try {
                    emitter.send(SseEmitter.event().name("token")
                            .data("{\"text\": \"The assistant is temporarily unavailable. Please try again.\"}"));
                    emitter.send(SseEmitter.event().name("done").data("{\"route\": \"error\", \"redacted\": []}"));
                    emitter.complete();
                } catch (Exception ignored) {
                    emitter.completeWithError(e);
                }
            }
        });
        return emitter;
    }

    @PostMapping("/documents")
    public ResponseEntity<Map<String, Object>> upload(@RequestParam("file") MultipartFile file) throws IOException {
        String name = file.getOriginalFilename();
        if (name == null || !(name.endsWith(".md") || name.endsWith(".txt"))) {
            return ResponseEntity.badRequest().body(Map.of("detail", "Only .md or .txt files"));
        }
        return ResponseEntity.ok(ai.upload(name, file.getBytes()));
    }
}
