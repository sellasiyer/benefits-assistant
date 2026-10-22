package com.sellasiyer.benefits.gateway;

import java.io.IOException;
import java.util.UUID;

import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Gives every request a correlation id (reusing the caller's X-Request-Id if present), puts it in the
 * logging MDC, echoes it in the response, and forwards it to the Python service, so one id traces a
 * request across both services' logs.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestIdFilter extends OncePerRequestFilter {

    public static final String HEADER = "X-Request-Id";
    public static final String ATTR = "requestId";

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse res, FilterChain chain)
            throws ServletException, IOException {
        String id = req.getHeader(HEADER);
        if (id == null || id.isBlank() || id.length() > 64) {
            id = UUID.randomUUID().toString().substring(0, 8);
        }
        req.setAttribute(ATTR, id);
        res.setHeader(HEADER, id);
        MDC.put(ATTR, id);
        try {
            chain.doFilter(req, res);
        } finally {
            MDC.remove(ATTR);
        }
    }
}
