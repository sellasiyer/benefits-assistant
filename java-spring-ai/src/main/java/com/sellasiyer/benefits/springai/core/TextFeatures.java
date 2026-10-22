package com.sellasiyer.benefits.springai.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Tokenizer shared by the offline embedding model and the offline answerer (mirrors the Python version). */
public final class TextFeatures {

    private static final Pattern WORD = Pattern.compile("[a-z0-9$%]+");
    private static final Set<String> STOP = Set.of((
            "a an the is are was were be to of and or in on for with what how do does i my me if "
            + "it this that by at as can much when within after before am have has").split(" "));

    private TextFeatures() {
    }

    public static List<String> tokens(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) {
            String t = m.group();
            if (STOP.contains(t)) continue;
            // crude stemming so "visits" ~ "visit", "drugs" ~ "drug"
            if (t.length() > 4 && t.endsWith("s")) t = t.substring(0, t.length() - 1);
            out.add(t);
        }
        return out;
    }
}
