package com.sellasiyer.benefits.springai;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

import com.sellasiyer.benefits.springai.core.ExtractiveAnswerer;

import reactor.core.publisher.Flux;

/**
 * A Spring AI {@link ChatModel} that runs with no API key. Implementing the interface ourselves is
 * also how you'd wrap an internal or self-hosted model so the rest of the app (ChatClient, advisors,
 * streaming) works unchanged. Real providers (OpenAI, Azure OpenAI, Anthropic) plug in the same way.
 */
public class OfflineChatModel implements ChatModel {

    private static final Pattern WORD = Pattern.compile("\\S+\\s*");

    @Override
    public ChatResponse call(Prompt prompt) {
        return response(ExtractiveAnswerer.answer(text(prompt)));
    }

    @Override
    public Flux<ChatResponse> stream(Prompt prompt) {
        String answer = ExtractiveAnswerer.answer(text(prompt));
        List<String> words = new ArrayList<>();
        Matcher m = WORD.matcher(answer);
        while (m.find()) words.add(m.group());
        return Flux.fromIterable(words).map(OfflineChatModel::response);
    }

    private static String text(Prompt prompt) {
        return prompt.getInstructions().stream().map(Message::getText).collect(Collectors.joining("\n"));
    }

    private static ChatResponse response(String text) {
        return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
    }
}
