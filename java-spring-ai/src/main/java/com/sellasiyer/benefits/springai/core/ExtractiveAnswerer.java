package com.sellasiyer.benefits.springai.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The "brain" of the offline chat model: answers by extracting the context sentences that best match
 * the question, weighting rare words higher (IDF-style). Stands in for a real LLM so the whole
 * pipeline (prompt, streaming, citations) runs without API keys.
 */
public final class ExtractiveAnswerer {

    public static final String DONT_KNOW = "I don't know based on the plan documents. Please contact member services.";
    private static final Pattern CONTEXT = Pattern.compile("<context>(.*?)</context>", Pattern.DOTALL);
    private static final Pattern QUESTION = Pattern.compile("Question:\\s*(.*)");
    private static final Pattern BLOCK = Pattern.compile("\\[(\\d+)\\](.*?)(?=\\n\\[\\d+\\]|\\z)", Pattern.DOTALL);
    private static final int MAX_SENTENCES = 2;

    private ExtractiveAnswerer() {
    }

    private record Sentence(String text, String citation, Set<String> tokens) {
    }

    public static String answer(String prompt) {
        Matcher ctx = CONTEXT.matcher(prompt);
        Matcher q = QUESTION.matcher(prompt);
        if (!ctx.find() || !q.find()) return "I don't have enough information to answer that.";

        Set<String> qTokens = new HashSet<>(TextFeatures.tokens(q.group(1)));
        List<Sentence> sentences = new ArrayList<>();
        Matcher block = BLOCK.matcher(ctx.group(1));
        while (block.find()) {
            String id = block.group(1);
            StringBuilder body = new StringBuilder();
            for (String line : block.group(2).split("\\R")) {
                if (!line.strip().startsWith("#")) body.append(line).append('\n');
            }
            for (String s : body.toString().strip().split("(?<=[.!?])\\s+")) {
                if (!s.isBlank()) sentences.add(new Sentence(s.strip(), id, new HashSet<>(TextFeatures.tokens(s))));
            }
        }
        Map<String, Integer> df = new HashMap<>();
        for (Sentence s : sentences) for (String t : s.tokens()) df.merge(t, 1, Integer::sum);

        record Scored(double score, Sentence sentence) {
        }
        List<Scored> scored = new ArrayList<>();
        for (Sentence s : sentences) {
            double score = 0;
            for (String t : qTokens) if (s.tokens().contains(t)) score += 1.0 / df.get(t);
            if (score > 0) scored.add(new Scored(score, s));
        }
        if (scored.isEmpty()) return DONT_KNOW;
        scored.sort((a, b) -> Double.compare(b.score(), a.score())); // stable: ties keep document order
        double top = scored.get(0).score();
        Set<String> picked = new LinkedHashSet<>();
        List<String> out = new ArrayList<>();
        for (Scored s : scored) {
            if (s.score() < top * 0.6 || out.size() == MAX_SENTENCES) break;
            if (picked.add(s.sentence().text())) out.add(s.sentence().text() + " [" + s.sentence().citation() + "]");
        }
        return String.join(" ", out);
    }
}
