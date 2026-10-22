package com.sellasiyer.benefits.springai.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Section-aware chunking: split on Markdown headings so a chunk never spans two sections, keep the
 * heading in the chunk and in metadata (for citations), and cap long sections at {@code maxChars}
 * with one sentence of overlap so a fact isn't cut in half at a boundary.
 */
public final class MarkdownChunker {

    public record Chunk(String source, String section, int index, String text) {
        public String id() {
            return source + "::" + index;
        }
    }

    private MarkdownChunker() {
    }

    public static List<Chunk> chunk(String markdown, String source, int maxChars) {
        List<Chunk> chunks = new ArrayList<>();
        String title = "";
        String section = "";
        StringBuilder current = new StringBuilder();
        for (String line : markdown.split("\\R")) {
            boolean h1 = line.startsWith("# ");
            boolean h2 = line.startsWith("## ");
            if (h1 || h2) {
                flush(chunks, source, section.isEmpty() ? title : section, current, maxChars);
                if (h1) {
                    title = line.substring(2).strip();
                    section = "";
                } else {
                    section = line.substring(3).strip();
                }
            }
            current.append(line).append('\n');
        }
        flush(chunks, source, section.isEmpty() ? title : section, current, maxChars);
        return chunks;
    }

    private static void flush(List<Chunk> out, String source, String section, StringBuilder buf, int maxChars) {
        String text = buf.toString().strip();
        buf.setLength(0);
        if (text.isEmpty()) return;
        if (text.length() <= maxChars) {
            out.add(new Chunk(source, section, out.size(), text));
            return;
        }
        // long section: pack sentences up to maxChars, carrying the last sentence over as overlap
        String heading = text.startsWith("#") ? text.substring(0, text.indexOf('\n') + 1) : "";
        String[] sentences = text.substring(heading.length()).strip().split("(?<=[.!?])\\s+");
        StringBuilder part = new StringBuilder(heading);
        String last = "";
        for (String s : sentences) {
            if (part.length() + s.length() > maxChars && part.length() > heading.length()) {
                out.add(new Chunk(source, section, out.size(), part.toString().strip()));
                part = new StringBuilder(heading).append(last).append(last.isEmpty() ? "" : " ");
            }
            part.append(s).append(' ');
            last = s;
        }
        if (part.length() > heading.length()) out.add(new Chunk(source, section, out.size(), part.toString().strip()));
    }
}
