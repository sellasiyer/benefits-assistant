package com.sellasiyer.benefits.springai;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/** Same endpoints and payloads as the Python FastAPI service, so the React UI works unchanged. */
@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    public record ChatRequest(@NotBlank @Size(min = 3, max = 2000) String question) {
    }

    public record ChatResponse(String request_id, String answer, String route, List<BenefitsAgent.Citation> citations,
                               List<String> redacted, List<String> trace, int latency_ms) {
    }

    private final BenefitsAgent agent;
    private final KnowledgeBase kb;
    private final AppProperties props;
    private final ObjectMapper json;

    public ChatController(BenefitsAgent agent, KnowledgeBase kb, AppProperties props, ObjectMapper json) {
        this.agent = agent;
        this.kb = kb;
        this.props = props;
        this.json = json;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("status", "ok");
        m.put("llm", props.llmProvider());
        m.put("embeddings", props.embeddingProvider());
        m.put("chunks", kb.chunkCount());
        m.put("backend", "java-spring-ai");
        return m;
    }

    @PostMapping("/ingest")
    public Map<String, Object> ingest() throws IOException {
        return Map.of("indexed", kb.ingestBundledDocs(), "chunks", kb.chunkCount());
    }

    @PostMapping("/documents")
    public ResponseEntity<Map<String, Object>> upload(@RequestParam("file") MultipartFile file) throws IOException {
        String name = file.getOriginalFilename();
        if (name == null || !(name.endsWith(".md") || name.endsWith(".txt"))) {
            return ResponseEntity.badRequest().body(Map.of("detail", "Only .md or .txt files"));
        }
        int n = kb.ingest(new String(file.getBytes(), StandardCharsets.UTF_8), name);
        return ResponseEntity.ok(Map.of("source", name, "chunks", n));
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest req) {
        String rid = UUID.randomUUID().toString().substring(0, 8);
        long t0 = System.nanoTime();
        BenefitsAgent.Result r = agent.run(req.question(), new BenefitsAgent.Listener() {
        });
        int ms = (int) ((System.nanoTime() - t0) / 1_000_000);
        // audit log: route and latency only; the question may contain PHI
        log.info("audit rid={} route={} ms={}", rid, r.route(), ms);
        return new ChatResponse(rid, r.answer(), r.route(), r.citations(), r.redacted(), r.trace(), ms);
    }

    @PostMapping("/chat/stream")
    public SseEmitter stream(@Valid @RequestBody ChatRequest req) {
        SseEmitter emitter = new SseEmitter(120_000L);
        Thread.startVirtualThread(() -> {
            try {
                boolean[] streamed = {false};
                BenefitsAgent.Result r = agent.run(req.question(), new BenefitsAgent.Listener() {
                    @Override
                    public void step(String node, String trace) {
                        send(emitter, "step", Map.of("node", node, "trace", List.of(trace)));
                    }

                    @Override
                    public void token(String text) {
                        streamed[0] = true;
                        send(emitter, "token", Map.of("text", text));
                    }

                    @Override
                    public boolean streaming() {
                        return true;
                    }
                });
                if (!streamed[0]) send(emitter, "token", Map.of("text", r.answer())); // tool / refusal answers arrive whole
                send(emitter, "citations", r.citations());
                send(emitter, "done", Map.of("route", r.route(), "redacted", r.redacted()));
                emitter.complete();
            } catch (Exception e) {
                log.warn("stream failed: {}", e.getMessage());
                emitter.completeWithError(e);
            }
        });
        return emitter;
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ResponseEntity<Map<String, String>> invalid(MethodArgumentNotValidException e) {
        String detail = e.getBindingResult().getFieldErrors().stream()
                .map(f -> f.getField() + ": " + f.getDefaultMessage()).collect(Collectors.joining("; "));
        return ResponseEntity.badRequest().body(Map.of("error", "validation_failed", "detail", detail));
    }

    private void send(SseEmitter emitter, String event, Object payload) {
        try {
            emitter.send(SseEmitter.event().name(event).data(json.writeValueAsString(payload)));
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException(e);
        } catch (IOException e) {
            throw new IllegalStateException("client disconnected", e);
        }
    }
}
