"""Tests for mn.redact — PII redaction from transcript segments."""

from mn.redact import redact, redact_text
from mn.transcribe import Segment


# -- redact_text() -----------------------------------------------------------


class TestRedactText:

    def test_phone_number_us(self):
        assert "[PHONE]" in redact_text("Call me at 555-123-4567.")

    def test_phone_number_with_parens(self):
        assert "[PHONE]" in redact_text("(555) 123-4567")

    def test_phone_number_with_country_code(self):
        assert "[PHONE]" in redact_text("+1 555-123-4567")

    def test_ssn(self):
        assert "[SSN]" in redact_text("My SSN is 123-45-6789.")

    def test_email(self):
        assert "[EMAIL]" in redact_text("Email me at john@example.com.")

    def test_date_slash(self):
        assert "[DATE]" in redact_text("Born on 01/15/1990.")

    def test_date_dash(self):
        assert "[DATE]" in redact_text("DOB: 01-15-1990")

    def test_address(self):
        result = redact_text("I live at 123 Main Street.")
        assert "[ADDRESS]" in result

    def test_address_abbreviated(self):
        result = redact_text("Office at 456 Oak Ave")
        assert "[ADDRESS]" in result

    def test_no_pii_unchanged(self):
        text = "I feel anxious about work."
        assert redact_text(text) == text

    def test_multiple_pii_in_one_string(self):
        result = redact_text("Call 555-1234567, email me@test.com, SSN 123-45-6789")
        assert "[PHONE]" in result
        assert "[EMAIL]" in result
        assert "[SSN]" in result

    def test_extra_names(self):
        result = redact_text("John told me about it.", extra_names=["John"])
        assert "[NAME]" in result
        assert "John" not in result

    def test_extra_names_case_insensitive(self):
        result = redact_text("john and JOHN were there.", extra_names=["John"])
        assert result.count("[NAME]") == 2

    def test_extra_names_multiple(self):
        result = redact_text(
            "John and Jane discussed it.",
            extra_names=["John", "Jane"],
        )
        assert result.count("[NAME]") == 2

    def test_extra_names_none(self):
        text = "Normal text."
        assert redact_text(text, extra_names=None) == text

    def test_extra_names_empty_list(self):
        text = "Normal text."
        assert redact_text(text, extra_names=[]) == text

    def test_extra_names_with_whitespace(self):
        result = redact_text("John was here.", extra_names=["  John  "])
        assert "[NAME]" in result


# -- redact() on Segments ---------------------------------------------------


class TestRedact:

    def test_redacts_segment_text(self):
        segs = [Segment("A", "Call 555-123-4567.", 0.0, 1.0)]
        result = redact(segs)
        assert "[PHONE]" in result[0].text

    def test_preserves_speaker_and_timestamps(self):
        segs = [Segment("Therapist", "Email me@test.com.", 1.5, 3.0)]
        result = redact(segs)
        assert result[0].speaker == "Therapist"
        assert result[0].start == 1.5
        assert result[0].end == 3.0

    def test_returns_new_list(self):
        segs = [Segment("A", "text", 0.0, 1.0)]
        result = redact(segs)
        assert result is not segs
        assert result[0] is not segs[0]

    def test_multiple_segments(self):
        segs = [
            Segment("A", "SSN is 123-45-6789.", 0.0, 1.0),
            Segment("B", "No PII here.", 1.0, 2.0),
        ]
        result = redact(segs)
        assert "[SSN]" in result[0].text
        assert result[1].text == "No PII here."

    def test_with_extra_names(self):
        segs = [Segment("A", "John feels anxious.", 0.0, 1.0)]
        result = redact(segs, extra_names=["John"])
        assert "[NAME]" in result[0].text

    def test_empty_segments(self):
        assert redact([]) == []
