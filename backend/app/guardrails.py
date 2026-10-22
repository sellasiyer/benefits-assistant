"""Input guardrails that run BEFORE anything reaches an LLM.

1. PHI/PII redaction - HIPAA: member identifiers never leave our boundary in a prompt.
2. Prompt-injection screen - reject obvious attempts to override the system prompt.
Production systems layer on: Azure AI Content Safety / Bedrock Guardrails, allow-listed
topics, output checks, and an audit log of every prompt and response.
"""
import re

PHI_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MEMBER_ID": re.compile(r"\b[A-Z]{2,3}\d{6,10}\b"),
    "DOB": re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(19|20)\d{2}\b"),
    "PHONE": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}

INJECTION = re.compile(
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard (the )?system prompt|you are now|reveal (your|the) (system )?prompt",
    re.I,
)


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in PHI_PATTERNS.items():
        if pat.search(text):
            found.append(label)
            text = pat.sub(f"[{label}]", text)
    return text, found


def is_injection(text: str) -> bool:
    return bool(INJECTION.search(text))
