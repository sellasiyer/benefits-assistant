package com.sellasiyer.benefits.springai;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Component;

import com.sellasiyer.benefits.springai.core.MarkdownChunker;

import jakarta.annotation.PostConstruct;

/**
 * RAG ingestion and retrieval: load → chunk → embed → store, then similarity search.
 * Stable chunk ids and delete-before-add make re-ingesting a document idempotent.
 */
@Component
public class KnowledgeBase {

    private static final Logger log = LoggerFactory.getLogger(KnowledgeBase.class);

    public record Hit(String source, String section, String text, double score) {
    }

    private final VectorStore store;
    private final AppProperties props;
    private final Map<String, List<String>> idsBySource = new ConcurrentHashMap<>();

    public KnowledgeBase(VectorStore store, AppProperties props) {
        this.store = store;
        this.props = props;
    }

    @PostConstruct
    void indexOnStartup() throws IOException {
        ingestBundledDocs();
    }

    public Map<String, Integer> ingestBundledDocs() throws IOException {
        Resource[] docs = new PathMatchingResourcePatternResolver().getResources("classpath:docs/*.md");
        Arrays.sort(docs, Comparator.comparing(Resource::getFilename));
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Resource r : docs) {
            counts.put(r.getFilename(), ingest(r.getContentAsString(StandardCharsets.UTF_8), r.getFilename()));
        }
        log.info("indexed {}", counts);
        return counts;
    }

    public synchronized int ingest(String markdown, String source) {
        List<String> old = idsBySource.remove(source);
        if (old != null && !old.isEmpty()) store.delete(old);
        List<Document> docs = MarkdownChunker.chunk(markdown, source, props.chunkSize()).stream()
                .map(c -> new Document(c.id(), c.text(),
                        Map.<String, Object>of("source", c.source(), "section", c.section(), "chunk", c.index())))
                .toList();
        store.add(docs);
        idsBySource.put(source, docs.stream().map(Document::getId).toList());
        return docs.size();
    }

    public List<Hit> retrieve(String query) {
        return store.similaritySearch(SearchRequest.builder().query(query).topK(props.topK()).build())
                .stream()
                .map(d -> new Hit(
                        String.valueOf(d.getMetadata().get("source")),
                        String.valueOf(d.getMetadata().getOrDefault("section", "")),
                        d.getText(),
                        d.getScore() == null ? 0.0 : Math.round(d.getScore() * 1000) / 1000.0))
                .toList();
    }

    public int chunkCount() {
        return idsBySource.values().stream().mapToInt(List::size).sum();
    }
}
