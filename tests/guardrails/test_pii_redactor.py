"""Unit tests for app/guardrails/pii_redactor.py."""

from __future__ import annotations

from app.guardrails.pii_redactor import DEFAULT_PII_REDACTOR, PIIRedactor


class TestPIIRedactor:
    """Tests for active PII scrubbing across strings and nested structures."""

    def test_redacts_ssn_email_phone_cc_and_dl_in_text(self) -> None:
        redactor = PIIRedactor()
        raw_note = (
            "Customer John Smith (SSN: 123-45-6789, DL: D99887766, "
            "email: john.smith@gmail.com, phone: (415) 555-0199) "
            "wants to put deposit on card 4532-1111-2222-3333."
        )
        scrubbed = redactor.redact_text(raw_note)

        assert "123-45-6789" not in scrubbed
        assert "D99887766" not in scrubbed
        assert "john.smith@gmail.com" not in scrubbed
        assert "(415) 555-0199" not in scrubbed
        assert "4532-1111-2222-3333" not in scrubbed

        assert "[REDACTED_SSN]" in scrubbed
        assert "[REDACTED_DRIVER_LICENSE]" in scrubbed
        assert "[REDACTED_EMAIL]" in scrubbed
        assert "[REDACTED_PHONE]" in scrubbed
        assert "[REDACTED_CREDIT_CARD]" in scrubbed

    def test_redacts_nested_dicts_lists_tuples_and_sensitive_keys(self) -> None:
        payload = {
            "customer_name": "Jane Roe",
            "ssn": "999001111",
            "contact": {
                "email": "jane.roe@company.org",
                "notes": ["Call back at 650-555-8899"],
                "phones_tuple": ("408-555-1234",),
            },
            "budget": 38000,
        }
        scrubbed = DEFAULT_PII_REDACTOR.redact_data(payload)

        assert scrubbed["ssn"] == "[REDACTED_SSN]"
        assert scrubbed["contact"]["email"] == "[REDACTED_EMAIL]"
        assert scrubbed["contact"]["notes"] == ["Call back at [REDACTED_PHONE]"]
        assert scrubbed["contact"]["phones_tuple"] == ("[REDACTED_PHONE]",)
        assert scrubbed["budget"] == 38000

    def test_contains_pii_detection(self) -> None:
        assert DEFAULT_PII_REDACTOR.contains_pii("Email me at a@b.com") is True
        assert DEFAULT_PII_REDACTOR.contains_pii("SSN 123-45-6789") is True
        assert DEFAULT_PII_REDACTOR.contains_pii("Looking for a hybrid SUV") is False
        assert DEFAULT_PII_REDACTOR.contains_pii("") is False
