"""Integration tests — real audio through the pipeline, no mocks.

These tests require: faster-whisper, pyannote.audio, numpy, soundfile.
They are skipped automatically if dependencies are missing or HF_TOKEN
is not set. Run explicitly with:

    HF_TOKEN=hf_... pytest tests/test_integration.py -v

The WAV fixture is generated programmatically — no checked-in binaries.
"""

import os
import struct
import tempfile

import pytest

# Skip the entire module if heavy deps aren't available.
whisper = pytest.importorskip("faster_whisper", reason="faster-whisper not installed")

has_hf_token = bool(os.environ.get("HF_TOKEN"))
skip_no_token = pytest.mark.skipif(
    not has_hf_token,
    reason="HF_TOKEN not set (required for pyannote diarization)",
)


def _make_silent_wav(path, duration=2.0, sample_rate=16000):
    """Generate a silent WAV file (pure stdlib, no numpy needed)."""
    n_samples = int(duration * sample_rate)
    data_size = n_samples * 2  # 16-bit mono

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))       # chunk size
        f.write(struct.pack("<H", 1))        # PCM
        f.write(struct.pack("<H", 1))        # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))        # block align
        f.write(struct.pack("<H", 16))       # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


# -- Whisper-only tests (no diarization, no HF_TOKEN needed) ----------------


class TestWhisperOnly:
    """Test transcription without diarization (no HF_TOKEN required)."""

    def test_transcribe_silent_wav(self, tmp_path):
        """Whisper should handle a silent WAV without crashing."""
        from mn.transcribe import transcribe

        wav = tmp_path / "silence.wav"
        _make_silent_wav(str(wav))

        words = transcribe(str(wav), model_size="tiny", device="cpu",
                           compute_type="int8")
        # Silent audio → few or no words. The key thing is it doesn't crash.
        assert isinstance(words, list)

    def test_align_produces_segments(self):
        """align() should produce valid Segments from fake word data."""
        from mn.transcribe import Segment, align

        words = [
            {"text": " Hello", "start": 0.0, "end": 0.5},
            {"text": " world", "start": 0.5, "end": 1.0},
        ]
        speakers = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
        ]
        result = align(words, speakers)
        assert len(result) == 1
        assert isinstance(result[0], Segment)
        assert "Hello" in result[0].text
        assert "world" in result[0].text


# -- Full pipeline tests (require HF_TOKEN) --------------------------------


@skip_no_token
class TestFullPipeline:
    """End-to-end tests that require pyannote diarization."""

    def test_transcribe_and_diarize_silent(self, tmp_path):
        """Full pipeline on silent audio — should not crash."""
        from mn.transcribe import transcribe_and_diarize

        wav = tmp_path / "silence.wav"
        _make_silent_wav(str(wav), duration=3.0)

        segments = transcribe_and_diarize(
            str(wav),
            model_size="tiny",
            device="cpu",
            compute_type="int8",
        )
        assert isinstance(segments, list)
        # All returned items should be Segments.
        from mn.transcribe import Segment
        for s in segments:
            assert isinstance(s, Segment)

    def test_serialization_roundtrip(self, tmp_path):
        """Transcribe → JSONL → parse back should preserve data."""
        from mn.transcribe import from_jsonl, to_jsonl, transcribe_and_diarize

        wav = tmp_path / "silence.wav"
        _make_silent_wav(str(wav))

        segments = transcribe_and_diarize(
            str(wav), model_size="tiny", device="cpu", compute_type="int8",
        )
        if segments:
            jsonl = to_jsonl(segments)
            restored = from_jsonl(jsonl)
            assert len(restored) == len(segments)
            for orig, rest in zip(segments, restored):
                assert orig.speaker == rest.speaker
                assert orig.text == rest.text
