"""Tests for mn.cli — argument parsing and pipeline wiring."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from mn import cli
from mn.transcribe import Segment, to_jsonl


def _sample_segments():
    return [
        Segment("SPEAKER_00", "I feel anxious.", 0.0, 1.2),
        Segment("SPEAKER_01", "Tell me more.", 1.5, 2.6),
    ]


def _sample_jsonl():
    return to_jsonl(_sample_segments())


# -- mn-fmt -----------------------------------------------------------------


class TestFmtCli:

    def test_basic_output(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", ["mn-fmt"])
        cli.fmt()
        out = capsys.readouterr().out
        assert "SPEAKER_00: I feel anxious." in out
        assert "SPEAKER_01: Tell me more." in out

    def test_timestamps_flag(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", ["mn-fmt", "--timestamps"])
        cli.fmt()
        out = capsys.readouterr().out
        assert "[00:00" in out
        assert "→" in out

    def test_empty_stdin_produces_no_output(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(""))
        monkeypatch.setattr("sys.argv", ["mn-fmt"])
        cli.fmt()
        out = capsys.readouterr().out
        assert out == ""

    def test_whitespace_only_stdin(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("  \n\n  "))
        monkeypatch.setattr("sys.argv", ["mn-fmt"])
        cli.fmt()
        out = capsys.readouterr().out
        assert out == ""


# -- mn-summarize -----------------------------------------------------------


class TestSummarizeCli:

    def test_calls_summarize_with_template(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Summarize: $transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--base-url", "http://test/v1",
            "--model", "test-model",
            "--api-key", "test-key",
        ])

        import httpx
        mock_resp = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Generated note"}}]},
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=mock_resp):
            cli.summarize()

        out = capsys.readouterr().out
        assert "Generated note" in out

    def test_empty_stdin_skips(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.stdin", StringIO(""))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
        ])
        cli.summarize()
        assert capsys.readouterr().out == ""

    def test_requires_template_flag(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", ["mn-summarize"])
        with pytest.raises(SystemExit):
            cli.summarize()


# -- mn-transcribe -----------------------------------------------------------


class TestTranscribeCli:

    def test_requires_audio_arg(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mn-transcribe"])
        with pytest.raises(SystemExit):
            cli.transcribe()

    def test_outputs_jsonl(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", "test.wav",
            "--num-speakers", "2",
        ])

        segments = _sample_segments()
        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=segments):
            cli.transcribe()

        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "speaker" in obj
            assert "text" in obj

    def test_passes_all_flags(self, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", "session.wav",
            "--model", "large-v3",
            "--device", "cuda",
            "--compute-type", "float16",
            "--num-speakers", "3",
            "--min-speakers", "2",
            "--max-speakers", "4",
            "--hf-token", "hf_test",
        ])

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=[]) as mock:
            cli.transcribe()
            mock.assert_called_once_with(
                "session.wav",
                model_size="large-v3",
                device="cuda",
                compute_type="float16",
                num_speakers=3,
                min_speakers=2,
                max_speakers=4,
                hf_token="hf_test",
            )


# -- mn (main) ---------------------------------------------------------------


class TestMainCli:

    def test_requires_audio_and_template(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mn"])
        with pytest.raises(SystemExit):
            cli.main()

    def test_transcript_only_mode(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn", "test.wav",
            "--template", str(template),
            "--transcript-only",
        ])

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=_sample_segments()):
            cli.main()

        out = capsys.readouterr().out
        assert "SPEAKER_00" in out
        assert "[00:00" in out  # transcript-only uses timestamps

    def test_full_pipeline(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Summarize: $transcript")

        monkeypatch.setattr("sys.argv", [
            "mn", "test.wav",
            "--template", str(template),
            "--base-url", "http://test/v1",
            "--llm-model", "test-model",
            "--api-key", "test-key",
        ])

        import httpx
        mock_resp = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Final note"}}]},
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=_sample_segments()):
            with patch("mn.summarize.httpx.post", return_value=mock_resp):
                cli.main()

        out = capsys.readouterr().out
        assert "Final note" in out

    def test_passes_whisper_and_diarization_flags(self, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn", "session.wav",
            "--template", str(template),
            "--transcript-only",
            "--model", "large-v3",
            "--device", "cuda",
            "--num-speakers", "2",
            "--hf-token", "hf_abc",
        ])

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=[]) as mock:
            cli.main()
            mock.assert_called_once_with(
                "session.wav",
                model_size="large-v3",
                device="cuda",
                compute_type="int8",
                num_speakers=2,
                min_speakers=None,
                max_speakers=None,
                hf_token="hf_abc",
            )
