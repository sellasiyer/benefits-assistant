package com.sellasiyer.benefits.springai.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Input guardrails that run BEFORE anything reaches an LLM, the vector store or the logs.
 * PHI/PII is masked (HIPAA: member identifiers never leave our boundary in a prompt), and obvious
 * prompt-injection attempts are refused. Production adds a content-safety service and output checks.
 */
public final class Guardrails {

    public record Redaction(String text, List<String> found) {
    }

    private static final Map<String, Pattern> PHI = new LinkedHashMap<>();

    static {
        PHI.put("SSN", Pattern.compile("\\b\\d{3}-\\d{2}-\\d{4}\\b"));
        PHI.put("MEMBER_ID", Pattern.compile("\\b[A-Z]{2,3}\\d{6,10}\\b"));
        PHI.put("DOB", Pattern.compile("\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\\d|3[01])/(19|20)\\d{2}\\b"));
        PHI.put("PHONE", Pattern.compile("\\b\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b"));
        PHI.put("EMAIL", Pattern.compile("\\b[\\w.+-]+@[\\w-]+\\.[\\w.]+\\b"));
    }

    private static final Pattern INJECTION = Pattern.compile(
            "ignore (all |the )?(previous|prior|above) (instructions|prompts?)"
                    + "|disregard (the )?system prompt|you are now|reveal (your|the) (system )?prompt",
            Pattern.CASE_INSENSITIVE);

    private Guardrails() {
    }

    public static Redaction redact(String text) {
        List<String> found = new ArrayList<>();
        for (var e : PHI.entrySet()) {
            if (e.getValue().matcher(text).find()) {
                found.add(e.getKey());
                text = e.getValue().matcher(text).replaceAll("[" + e.getKey() + "]");
            }
        }
        return new Redaction(text, found);
    }

    public static boolean isInjection(String text) {
        return INJECTION.matcher(text).find();
    }
}
