"""Tests for mn.edit — round-trip editable transcript format."""

from mn.edit import _parse_time, from_editable, to_editable
from mn.transcribe import Segment


def _sample():
    return [
        Segment("SPEAKER_00", "I've been feeling anxious.", 0.0, 65.0),
        Segment("SPEAKER_01", "Can you tell me more?", 70.0, 125.0),
    ]


# -- to_editable() ----------------------------------------------------------


class TestToEditable:

    def test_basic_format(self):
        result = to_editable(_sample())
        assert "[00:00 → 01:05] SPEAKER_00:" in result
        assert "I've been feeling anxious." in result
        assert "[01:10 → 02:05] SPEAKER_01:" in result
        assert "Can you tell me more?" in result

    def test_blocks_separated_by_blank_lines(self):
        result = to_editable(_sample())
        blocks = result.strip().split("\n\n")
        assert len(blocks) == 2

    def test_ends_with_newline(self):
        result = to_editable(_sample())
        assert result.endswith("\n")

    def test_single_segment(self):
        result = to_editable([Segment("A", "hello", 0.0, 5.0)])
        assert "[00:00 → 00:05] A:" in result
        assert "hello" in result

    def test_empty(self):
        assert to_editable([]) == "\n"


# -- from_editable() --------------------------------------------------------


class TestFromEditable:

    def test_parses_basic(self):
        text = (
            "[00:00 → 01:05] SPEAKER_00:\n"
            "I've been feeling anxious.\n"
            "\n"
            "[01:10 → 02:05] SPEAKER_01:\n"
            "Can you tell me more?\n"
        )
        result = from_editable(text)
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[0].text == "I've been feeling anxious."
        assert result[0].start == 0.0
        assert result[0].end == 65.0
        assert result[1].speaker == "SPEAKER_01"
        assert result[1].start == 70.0
        assert result[1].end == 125.0

    def test_preserves_edited_speaker_names(self):
        text = (
            "[00:00 → 00:05] Therapist:\n"
            "How are you?\n"
        )
        result = from_editable(text)
        assert result[0].speaker == "Therapist"

    def test_preserves_edited_text(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "Corrected transcription text.\n"
        )
        result = from_editable(text)
        assert result[0].text == "Corrected transcription text."

    def test_multiline_body(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "First line.\n"
            "Second line.\n"
        )
        result = from_editable(text)
        assert "First line.\nSecond line." == result[0].text

    def test_empty_string(self):
        assert from_editable("") == []

    def test_skips_malformed_blocks(self):
        text = (
            "This is not a valid header\n"
            "some text\n"
            "\n"
            "[00:00 → 00:05] A:\n"
            "Valid block.\n"
        )
        result = from_editable(text)
        assert len(result) == 1
        assert result[0].speaker == "A"


# -- Round-trip -------------------------------------------------------------


class TestRoundTrip:

    def test_roundtrip_preserves_content(self):
        original = _sample()
        text = to_editable(original)
        restored = from_editable(text)
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig.speaker == rest.speaker
            assert orig.text == rest.text
            # Timestamps lose sub-second precision through MM:SS format
            assert abs(orig.start - rest.start) < 60
            assert abs(orig.end - rest.end) < 60

    def test_roundtrip_single_segment(self):
        original = [Segment("Solo", "just me talking", 30.0, 90.0)]
        restored = from_editable(to_editable(original))
        assert restored[0].speaker == "Solo"
        assert restored[0].text == "just me talking"


# -- _parse_time() ----------------------------------------------------------


class TestParseTime:

    def test_zero(self):
        assert _parse_time("0:00") == 0

    def test_seconds(self):
        assert _parse_time("0:45") == 45

    def test_minutes_and_seconds(self):
        assert _parse_time("2:05") == 125

    def test_large_minutes(self):
        assert _parse_time("61:01") == 3661
