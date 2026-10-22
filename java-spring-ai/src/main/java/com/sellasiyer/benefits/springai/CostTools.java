package com.sellasiyer.benefits.springai;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import com.sellasiyer.benefits.springai.core.CostMath;

/**
 * Tools the LLM can call. Spring AI turns the {@code @Tool} method into a JSON schema the model sees;
 * when the model asks for the tool, Spring AI invokes this method and feeds the result back.
 * Offline, the agent calls it directly.
 */
@Component
public class CostTools {

    private final CostMath math;

    public CostTools(CostMath math) {
        this.math = math;
    }

    @Tool(description = "Estimate what a member pays in network for one service under the Gold PPO plan")
    public CostMath.Estimate estimateVisitCost(
            @ToolParam(description = "One of primary_care, specialist, urgent_care, emergency_room, procedure") String service,
            @ToolParam(description = "Billed amount in US dollars") double billedAmount,
            @ToolParam(description = "Deductible still remaining in US dollars; 0 if already met") double deductibleRemaining) {
        return math.estimate(service, billedAmount, deductibleRemaining);
    }
}
