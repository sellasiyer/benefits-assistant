package com.sellasiyer.benefits.springai.core;

import java.util.Map;

/** Deterministic plan math. LLMs are unreliable at arithmetic, so the model calls this as a tool. */
public final class CostMath {

    public record Estimate(double memberPays, double planPays, String explanation) {
    }

    private final Map<String, Double> copays;
    private final double coinsurance;
    private final double oopMax;

    public CostMath(Map<String, Double> copays, double coinsurance, double oopMax) {
        this.copays = copays;
        this.coinsurance = coinsurance;
        this.oopMax = oopMax;
    }

    public Estimate estimate(String service, double billedAmount, double deductibleRemaining) {
        double pay;
        String how;
        if (copays.containsKey(service)) {
            double copay = copays.get(service);
            pay = Math.min(copay, billedAmount);
            how = String.format("flat $%.0f copay for %s", copay, service.replace('_', ' '));
        } else {
            double ded = Math.min(billedAmount, deductibleRemaining);
            double coins = (billedAmount - ded) * coinsurance;
            pay = Math.min(ded + coins, oopMax);
            how = String.format("$%,.2f toward the deductible + %.0f%% coinsurance on the remaining $%,.2f ($%,.2f)",
                    ded, coinsurance * 100, billedAmount - ded, coins);
        }
        pay = Math.round(pay * 100) / 100.0;
        return new Estimate(pay, Math.round((billedAmount - pay) * 100) / 100.0, how);
    }
}
