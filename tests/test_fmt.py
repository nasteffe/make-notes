"""Tests for mn.fmt — text formatting of transcript segments."""

from mn.fmt import _ftime, fmt
from mn.transcribe import Segment


def _two_segments():
    return [
        Segment("SPEAKER_00", "I've been feeling anxious.", 0.0, 1.2),
        Segment("SPEAKER_01", "Can you tell me more?", 1.5, 2.6),
    ]


# -- fmt() ------------------------------------------------------------------


class TestFmt:

    def test_basic_format(self):
        result = fmt(_two_segments())
        assert "SPEAKER_00: I've been feeling anxious." in result
        assert "SPEAKER_01: Can you tell me more?" in result

    def test_segments_separated_by_blank_line(self):
        result = fmt(_two_segments())
        lines = result.split("\n\n")
        assert len(lines) == 2

    def test_no_timestamps_by_default(self):
        result = fmt(_two_segments())
        assert "[" not in result
        assert "→" not in result

    def test_timestamps_included_when_requested(self):
        result = fmt(_two_segments(), timestamps=True)
        assert "[00:00 → 00:01]" in result
        assert "[00:01 → 00:02]" in result

    def test_timestamp_format(self):
        result = fmt(_two_segments(), timestamps=True)
        # First segment: 0.0 → 1.2 = 00:00 → 00:01
        assert result.startswith("[00:00 → 00:01] SPEAKER_00:")

    def test_single_segment(self):
        seg = [Segment("A", "hello", 0.0, 1.0)]
        result = fmt(seg)
        assert result == "A: hello"

    def test_empty_segments(self):
        assert fmt([]) == ""

    def test_long_timestamps(self):
        seg = [Segment("A", "late", 3661.0, 3725.0)]  # 61:01 → 62:05
        result = fmt(seg, timestamps=True)
        assert "[61:01 → 62:05]" in result

    def test_three_speakers(self):
        segs = [
            Segment("Client", "I feel better.", 0.0, 1.0),
            Segment("Therapist", "That's great.", 1.0, 2.0),
            Segment("Supervisor", "Agreed.", 2.0, 3.0),
        ]
        result = fmt(segs)
        assert result.count(":") == 3  # each speaker line has one colon


# -- _ftime() ---------------------------------------------------------------


class TestFtime:

    def test_zero(self):
        assert _ftime(0) == "00:00"

    def test_seconds_only(self):
        assert _ftime(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert _ftime(125) == "02:05"

    def test_truncates_fractional(self):
        assert _ftime(1.9) == "00:01"

    def test_large_value(self):
        assert _ftime(3600) == "60:00"  # 1 hour = 60 minutes
