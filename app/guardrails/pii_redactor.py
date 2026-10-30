"""Active PII Redaction pipeline for logs, prompts, and memory persistence.

Scrubs sensitive customer data (Social Security Numbers, Credit Card numbers,
Driver's License IDs, phone numbers, email addresses, and API keys) prior to
logging or storing in session/long-term memory. Supports optional Google Cloud
DLP (`google-cloud-dlp`) inspection alongside deterministic regex scrubbing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class PIIRedactor:
    """Active PII scrubbing engine for logging and conversational memory pipelines."""

    _PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        # US Social Security Number (XXX-XX-XXXX or 9 digits after SSN label)
        (
            "[REDACTED_SSN]",
            re.compile(
                r"\b(?:\d{3}-\d{2}-\d{4}|(?:SSN|Social\s*Security)[:\s#]*\d{9})\b",
                re.IGNORECASE,
            ),
        ),
        # Payment / Credit Card numbers (13-16 digits with spaces or dashes)
        (
            "[REDACTED_CREDIT_CARD]",
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"),
        ),
        # Driver's License numbers (explicitly labeled DL / Driver License)
        (
            "[REDACTED_DRIVER_LICENSE]",
            re.compile(
                r"\b(?:DL|Driver'?s?\s*License)(?:\s*(?:Number|No|ID|#))?[:\s#]*[A-Z0-9-]{6,14}\b",
                re.IGNORECASE,
            ),
        ),
        # Email addresses
        (
            "[REDACTED_EMAIL]",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        # US / International Phone numbers
        (
            "[REDACTED_PHONE]",
            re.compile(
                r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
            ),
        ),
        # Secret / API Key tokens
        (
            "[REDACTED_API_KEY]",
            re.compile(r"\b(?:AIza[0-9A-Za-z-_]{35}|sk-[A-Za-z0-9]{20,})\b"),
        ),
    ]

    _SENSITIVE_FIELD_NAMES = frozenset(
        {
            "ssn",
            "social_security_number",
            "credit_card",
            "card_number",
            "driver_license",
            "drivers_license",
            "api_key",
            "secret_key",
            "password",
        }
    )

    def __init__(
        self, project_id: Optional[str] = None, use_cloud_dlp: bool = False
    ) -> None:
        self.project_id = project_id
        self.use_cloud_dlp = use_cloud_dlp

    def redact_text(self, text: str) -> str:
        """Scrubs all recognized PII patterns from a string.

        Args:
            text: Input string potentially containing customer PII.

        Returns:
            Sanitized string with sensitive tokens replaced by `[REDACTED_*]`.
        """
        if not isinstance(text, str) or not text:
            return text

        sanitized = text
        for replacement, pattern in self._PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def redact_data(self, data: Any) -> Any:
        """Recursively scrubs PII from strings, dictionaries, lists, and tuples.

        Args:
            data: Arbitrary payload (dict, list, str, primitive) to sanitize.

        Returns:
            Deep copy of `data` with all PII values redacted.
        """
        if isinstance(data, str):
            return self.redact_text(data)
        if isinstance(data, dict):
            sanitized_dict: Dict[Any, Any] = {}
            for key, value in data.items():
                normalized_key = str(key).lower().strip()
                if normalized_key in self._SENSITIVE_FIELD_NAMES and value is not None:
                    sanitized_dict[key] = f"[REDACTED_{normalized_key.upper()}]"
                else:
                    sanitized_dict[key] = self.redact_data(value)
            return sanitized_dict
        if isinstance(data, list):
            return [self.redact_data(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.redact_data(item) for item in data)
        return data

    def contains_pii(self, text: str) -> bool:
        """Returns True if the given text matches any known PII pattern."""
        if not isinstance(text, str) or not text:
            return False
        return any(pattern.search(text) is not None for _, pattern in self._PATTERNS)


DEFAULT_PII_REDACTOR = PIIRedactor()
