package com.sellasiyer.benefits.springai.core;

import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Where a question should go. With a real LLM this record is the structured-output schema
 * (ChatClient ... .entity(RouteDecision.class)); offline, {@link #heuristic(String)} fills it.
 *
 * @param route          "rag" or "calculator"
 * @param service        primary_care | specialist | urgent_care | emergency_room | procedure
 * @param billedAmount   dollar amount in the question, if any
 * @param deductibleMet  whether the member says they've met the deductible
 */
public record RouteDecision(String route, String service, double billedAmount, boolean deductibleMet) {

    private static final Pattern AMOUNT = Pattern.compile("\\$\\s?([\\d,]+(?:\\.\\d+)?)");
    private static final Pattern ASKS_COST = Pattern.compile("\\b(pay|owe|cost|charge|bill|out of pocket)", Pattern.CASE_INSENSITIVE);
    private static final Pattern MET = Pattern.compile("(met|hit|reached|satisfied) (my |the )?deductible", Pattern.CASE_INSENSITIVE);
    private static final Pattern NOT_MET = Pattern.compile("(not|n't|never) (yet )?(met|hit|reached|satisfied)", Pattern.CASE_INSENSITIVE);
    private static final Map<String, String[]> SERVICE_WORDS = new java.util.LinkedHashMap<>();

    static {
        SERVICE_WORDS.put("emergency_room", new String[] {"emergency", " er "});
        SERVICE_WORDS.put("urgent_care", new String[] {"urgent"});
        SERVICE_WORDS.put("specialist", new String[] {"specialist", "cardiolog", "dermatolog", "orthoped"});
        SERVICE_WORDS.put("primary_care", new String[] {"primary care", "pcp", "family doctor", "physical"});
    }

    public static RouteDecision rag() {
        return new RouteDecision("rag", "procedure", 0, false);
    }

    public static RouteDecision heuristic(String question) {
        Matcher amt = AMOUNT.matcher(question);
        if (!amt.find() || !ASKS_COST.matcher(question).find()) return rag();
        String low = " " + question.toLowerCase(Locale.ROOT) + " ";
        String service = "procedure";
        outer:
        for (var e : SERVICE_WORDS.entrySet()) {
            for (String w : e.getValue()) {
                if (low.contains(w)) {
                    service = e.getKey();
                    break outer;
                }
            }
        }
        boolean met = MET.matcher(question).find() && !NOT_MET.matcher(question).find();
        return new RouteDecision("calculator", service, Double.parseDouble(amt.group(1).replace(",", "")), met);
    }
}
