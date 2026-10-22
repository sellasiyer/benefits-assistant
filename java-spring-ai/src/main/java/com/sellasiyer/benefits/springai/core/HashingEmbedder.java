package com.sellasiyer.benefits.springai.core;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/**
 * Feature-hashed bag of words + bigrams, L2-normalised. Deterministic and offline, so the RAG
 * pipeline runs with no API keys. Same algorithm as the Python service's HashingEmbeddings.
 * A real embedding model (OpenAI, Azure OpenAI, Bedrock Titan) understands meaning; this does not.
 */
public final class HashingEmbedder {

    public static final int DIM = 768;
    private static final BigInteger BIG_DIM = BigInteger.valueOf(DIM);

    private HashingEmbedder() {
    }

    public static float[] embed(String text) {
        double[] vec = new double[DIM];
        List<String> toks = TextFeatures.tokens(text == null ? "" : text);
        List<String> feats = new ArrayList<>(toks);
        for (int i = 0; i + 1 < toks.size(); i++) feats.add(toks.get(i) + "_" + toks.get(i + 1));
        for (String f : feats) {
            BigInteger h = new BigInteger(1, md5(f));
            int bucket = h.mod(BIG_DIM).intValue();
            boolean positive = h.shiftRight(64).testBit(0);
            vec[bucket] += positive ? 1.0 : -1.0;
        }
        double norm = 0;
        for (double v : vec) norm += v * v;
        norm = norm == 0 ? 1.0 : Math.sqrt(norm);
        float[] out = new float[DIM];
        for (int i = 0; i < DIM; i++) out[i] = (float) (vec[i] / norm);
        return out;
    }

    public static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return (na == 0 || nb == 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private static byte[] md5(String s) {
        try {
            return MessageDigest.getInstance("MD5").digest(s.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
