package com.sellasiyer.benefits.gateway;

import java.util.stream.Collectors;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.HttpClientErrorException;

import com.sellasiyer.benefits.gateway.Dtos.ErrorResponse;

import jakarta.servlet.http.HttpServletRequest;

/** Consistent JSON errors with the request id, instead of stack traces. */
@RestControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ResponseEntity<ErrorResponse> invalid(MethodArgumentNotValidException e, HttpServletRequest req) {
        String detail = e.getBindingResult().getFieldErrors().stream()
                .map(f -> f.getField() + ": " + f.getDefaultMessage())
                .collect(Collectors.joining("; "));
        return ResponseEntity.badRequest().body(new ErrorResponse("validation_failed", detail, id(req)));
    }

    @ExceptionHandler(AiServiceUnavailableException.class)
    ResponseEntity<ErrorResponse> unavailable(AiServiceUnavailableException e, HttpServletRequest req) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(new ErrorResponse("ai_service_unavailable", e.getMessage(), id(req)));
    }

    @ExceptionHandler(HttpClientErrorException.class)
    ResponseEntity<ErrorResponse> upstream4xx(HttpClientErrorException e, HttpServletRequest req) {
        return ResponseEntity.status(e.getStatusCode())
                .body(new ErrorResponse("rejected_by_ai_service", e.getResponseBodyAsString(), id(req)));
    }

    private static String id(HttpServletRequest req) {
        Object id = req.getAttribute(RequestIdFilter.ATTR);
        return id == null ? null : id.toString();
    }
}
