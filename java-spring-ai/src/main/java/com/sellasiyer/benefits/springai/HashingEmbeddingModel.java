package com.sellasiyer.benefits.springai;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

import com.sellasiyer.benefits.springai.core.HashingEmbedder;

/** A Spring AI {@link EmbeddingModel} backed by deterministic feature hashing: no network, no key. */
public class HashingEmbeddingModel implements EmbeddingModel {

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        List<Embedding> out = new ArrayList<>();
        List<String> inputs = request.getInstructions();
        for (int i = 0; i < inputs.size(); i++) {
            out.add(new Embedding(HashingEmbedder.embed(inputs.get(i)), i));
        }
        return new EmbeddingResponse(out);
    }

    @Override
    public float[] embed(Document document) {
        return HashingEmbedder.embed(document.getText());
    }

    @Override
    public int dimensions() {
        return HashingEmbedder.DIM;
    }
}
