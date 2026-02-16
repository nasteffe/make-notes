"""Tests for mn.transcribe — alignment, serialization, and edge cases."""

import json

from mn.transcribe import Segment, align, from_jsonl, to_jsonl


# -- Fixtures ---------------------------------------------------------------

def _words_two_speakers():
    """Simulated whisper output: two speakers in a therapy session."""
    return [
        {"text": " I've", "start": 0.0, "end": 0.3},
        {"text": " been", "start": 0.3, "end": 0.5},
        {"text": " feeling", "start": 0.5, "end": 0.8},
        {"text": " anxious.", "start": 0.8, "end": 1.2},
        {"text": " Can", "start": 1.5, "end": 1.7},
        {"text": " you", "start": 1.7, "end": 1.9},
        {"text": " tell", "start": 1.9, "end": 2.1},
        {"text": " me", "start": 2.1, "end": 2.3},
        {"text": " more?", "start": 2.3, "end": 2.6},
    ]


def _speakers_two():
    """Simulated pyannote output: two speaker segments."""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.3},
        {"speaker": "SPEAKER_01", "start": 1.4, "end": 2.7},
    ]


def _sample_segments():
    """Pre-built Segment objects for serialization tests."""
    return [
        Segment("SPEAKER_00", "I've been feeling anxious.", 0.0, 1.2),
        Segment("SPEAKER_01", "Can you tell me more?", 1.5, 2.6),
    ]


# -- align() ----------------------------------------------------------------


class TestAlign:

    def test_two_speakers_basic(self):
        result = align(_words_two_speakers(), _speakers_two())
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_text_is_stripped_and_merged(self):
        result = align(_words_two_speakers(), _speakers_two())
        assert "anxious." in result[0].text
        assert "more?" in result[1].text

    def test_timestamps_span_word_boundaries(self):
        result = align(_words_two_speakers(), _speakers_two())
        assert result[0].start == 0.0
        assert result[0].end == 1.2
        assert result[1].start == 1.5
        assert result[1].end == 2.6

    def test_empty_words_returns_empty(self):
        assert align([], _speakers_two()) == []

    def test_no_speakers_falls_back_to_single(self):
        words = [
            {"text": " hello", "start": 0.0, "end": 0.5},
            {"text": " world", "start": 0.5, "end": 1.0},
        ]
        result = align(words, [])
        assert len(result) == 1
        assert result[0].speaker == "Speaker"
        assert result[0].text == "hello world"

    def test_single_word(self):
        words = [{"text": " hi", "start": 0.0, "end": 0.3}]
        speakers = [{"speaker": "A", "start": 0.0, "end": 1.0}]
        result = align(words, speakers)
        assert len(result) == 1
        assert result[0].speaker == "A"
        assert result[0].text == "hi"

    def test_three_speakers(self):
        words = [
            {"text": " one", "start": 0.0, "end": 0.5},
            {"text": " two", "start": 1.0, "end": 1.5},
            {"text": " three", "start": 2.0, "end": 2.5},
        ]
        speakers = [
            {"speaker": "A", "start": 0.0, "end": 0.6},
            {"speaker": "B", "start": 0.9, "end": 1.6},
            {"speaker": "C", "start": 1.9, "end": 2.6},
        ]
        result = align(words, speakers)
        assert len(result) == 3
        assert [s.speaker for s in result] == ["A", "B", "C"]

    def test_consecutive_words_same_speaker_merge(self):
        words = [
            {"text": " a", "start": 0.0, "end": 0.2},
            {"text": " b", "start": 0.2, "end": 0.4},
            {"text": " c", "start": 0.4, "end": 0.6},
        ]
        speakers = [{"speaker": "X", "start": 0.0, "end": 1.0}]
        result = align(words, speakers)
        assert len(result) == 1
        assert result[0].text == "a b c"

    def test_speaker_alternation_creates_separate_segments(self):
        words = [
            {"text": " yes", "start": 0.0, "end": 0.3},
            {"text": " no", "start": 0.5, "end": 0.8},
            {"text": " maybe", "start": 1.0, "end": 1.3},
        ]
        speakers = [
            {"speaker": "A", "start": 0.0, "end": 0.4},
            {"speaker": "B", "start": 0.4, "end": 0.9},
            {"speaker": "A", "start": 0.9, "end": 1.4},
        ]
        result = align(words, speakers)
        assert len(result) == 3
        assert [s.speaker for s in result] == ["A", "B", "A"]

    def test_word_outside_all_speaker_segments(self):
        """Word that falls in a gap between speaker segments."""
        words = [{"text": " gap", "start": 5.0, "end": 5.5}]
        speakers = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "B", "start": 10.0, "end": 11.0},
        ]
        result = align(words, speakers)
        assert len(result) == 1
        # Falls back to "Unknown" since no overlap and midpoint not in any segment
        assert result[0].speaker == "Unknown"

    def test_word_midpoint_fallback(self):
        """Word with zero overlap but midpoint inside a speaker segment."""
        words = [{"text": " hi", "start": 0.5, "end": 0.5}]  # zero-length
        speakers = [{"speaker": "A", "start": 0.0, "end": 1.0}]
        result = align(words, speakers)
        assert result[0].speaker == "A"

    def test_overlap_prefers_largest(self):
        """Word that overlaps two speakers — pick the one with more overlap."""
        words = [{"text": " split", "start": 0.8, "end": 1.5}]
        speakers = [
            {"speaker": "A", "start": 0.0, "end": 1.0},  # overlap: 0.2
            {"speaker": "B", "start": 1.0, "end": 2.0},  # overlap: 0.5
        ]
        result = align(words, speakers)
        assert result[0].speaker == "B"


# -- Serialization ----------------------------------------------------------


class TestSerialization:

    def test_to_jsonl_format(self):
        segments = _sample_segments()
        jsonl = to_jsonl(segments)
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert set(obj.keys()) == {"speaker", "text", "start", "end"}

    def test_roundtrip(self):
        original = _sample_segments()
        jsonl = to_jsonl(original)
        restored = from_jsonl(jsonl)
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig.speaker == rest.speaker
            assert orig.text == rest.text
            assert orig.start == rest.start
            assert orig.end == rest.end

    def test_from_jsonl_ignores_blank_lines(self):
        jsonl = (
            '{"speaker": "A", "text": "hello", "start": 0.0, "end": 1.0}\n'
            "\n"
            '{"speaker": "B", "text": "world", "start": 1.0, "end": 2.0}\n'
            "\n"
        )
        result = from_jsonl(jsonl)
        assert len(result) == 2

    def test_single_segment_roundtrip(self):
        original = [Segment("Solo", "just me", 0.0, 5.0)]
        assert from_jsonl(to_jsonl(original)) == original

    def test_to_jsonl_empty(self):
        assert to_jsonl([]) == ""

    def test_from_jsonl_preserves_float_precision(self):
        seg = Segment("A", "precise", 1.123456, 2.654321)
        restored = from_jsonl(to_jsonl([seg]))[0]
        assert restored.start == 1.123456
        assert restored.end == 2.654321


# -- Segment dataclass ------------------------------------------------------


class TestSegment:

    def test_equality(self):
        a = Segment("A", "hello", 0.0, 1.0)
        b = Segment("A", "hello", 0.0, 1.0)
        assert a == b

    def test_inequality(self):
        a = Segment("A", "hello", 0.0, 1.0)
        b = Segment("B", "hello", 0.0, 1.0)
        assert a != b

    def test_fields(self):
        s = Segment("X", "test", 1.5, 3.0)
        assert s.speaker == "X"
        assert s.text == "test"
        assert s.start == 1.5
        assert s.end == 3.0
