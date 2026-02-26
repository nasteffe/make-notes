"""Redact personally identifiable information from transcript segments.

Regex-based PII detection for common patterns. Runs as a filter:

    mn-transcribe session.wav | mn-redact | mn-summarize --template ...

Redacts: phone numbers, SSNs, email addresses, dates of birth,
street addresses (heuristic), and names (when provided via --names).
"""

import re

from .transcribe import Segment


# -- Patterns ---------------------------------------------------------------

_PATTERNS = [
    # US phone numbers
    (re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ), "[PHONE]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # Email
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Date patterns (MM/DD/YYYY, MM-DD-YYYY, etc.)
    (re.compile(
        r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    ), "[DATE]"),
    # Street addresses (number + street name + suffix)
    (re.compile(
        r"\b\d{1,5}\s+[\w\s]{1,30}"
        r"(?:street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln"
        r"|road|rd|court|ct|place|pl|way|circle|cir)\b",
        re.IGNORECASE,
    ), "[ADDRESS]"),
]


# -- Core -------------------------------------------------------------------


def redact_text(text, extra_names=None):
    """Apply PII patterns to a string, returning redacted version."""
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)

    # Redact specific names if provided.
    if extra_names:
        for name in extra_names:
            name = name.strip()
            if name:
                result = re.sub(
                    r"\b" + re.escape(name) + r"\b",
                    "[NAME]",
                    result,
                    flags=re.IGNORECASE,
                )
    return result


def redact(segments, extra_names=None):
    """Redact PII from a list of Segments. Returns new list."""
    return [
        Segment(s.speaker, redact_text(s.text, extra_names), s.start, s.end)
        for s in segments
    ]
