"""Tests for mn.cli — argument parsing and pipeline wiring."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import httpx
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


def _mock_llm_response(content="Generated note"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", "http://test/v1/chat/completions"),
    )


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


# -- mn-redact --------------------------------------------------------------


class TestRedactCli:

    def test_basic_redaction(self, capsys, monkeypatch):
        segs = [Segment("A", "Call 555-123-4567.", 0.0, 1.0)]
        monkeypatch.setattr("sys.stdin", StringIO(to_jsonl(segs)))
        monkeypatch.setattr("sys.argv", ["mn-redact"])
        cli.redact()
        out = capsys.readouterr().out
        obj = json.loads(out.strip())
        assert "[PHONE]" in obj["text"]

    def test_names_flag(self, capsys, monkeypatch):
        segs = [Segment("A", "John feels anxious.", 0.0, 1.0)]
        monkeypatch.setattr("sys.stdin", StringIO(to_jsonl(segs)))
        monkeypatch.setattr("sys.argv", ["mn-redact", "--names", "John"])
        cli.redact()
        out = capsys.readouterr().out
        obj = json.loads(out.strip())
        assert "[NAME]" in obj["text"]

    def test_empty_stdin(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(""))
        monkeypatch.setattr("sys.argv", ["mn-redact"])
        cli.redact()
        assert capsys.readouterr().out == ""


# -- mn-summarize -----------------------------------------------------------


class TestSummarizeCli:

    def test_calls_summarize_with_template(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Summarize: $transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--base-url", "http://test/v1",
            "--llm-model", "test-model",
            "--api-key", "test-key",
        ])

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_llm_response()):
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

    def test_redact_flag(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        segs = [Segment("A", "Call 555-123-4567.", 0.0, 1.0)]
        monkeypatch.setattr("sys.stdin", StringIO(to_jsonl(segs)))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--redact",
            "--base-url", "http://test/v1",
            "--llm-model", "m", "--api-key", "k",
        ])

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_llm_response("OK")) as mock:
            cli.summarize()
            prompt = mock.call_args[1]["json"]["messages"][0]["content"]
            assert "[PHONE]" in prompt
            assert "555-123-4567" not in prompt

    def test_client_name_and_date(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Client: $client_name Date: $date\n$transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--client-name", "J.D.",
            "--session-date", "2026-02-16",
            "--base-url", "http://test/v1",
            "--llm-model", "m", "--api-key", "k",
        ])

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_llm_response("OK")) as mock:
            cli.summarize()
            prompt = mock.call_args[1]["json"]["messages"][0]["content"]
            assert "J.D." in prompt
            assert "2026-02-16" in prompt


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

    def test_speakers_flag_relabels(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", "test.wav",
            "--speakers", "Therapist,Client",
        ])

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=_sample_segments()):
            cli.transcribe()

        out = capsys.readouterr().out
        lines = [json.loads(l) for l in out.strip().split("\n")]
        assert lines[0]["speaker"] == "Therapist"
        assert lines[1]["speaker"] == "Client"

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
                _whisper=None,
                _diarizer=None,
            )


# -- mn-templates -----------------------------------------------------------


class TestTemplatesCli:

    def test_lists_builtin_templates(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mn-templates"])
        cli.templates()
        out = capsys.readouterr().out
        assert "soap" in out
        assert "dap" in out
        assert "birp" in out
        assert "progress" in out
        assert "cbt-soap" in out
        assert "psychodynamic" in out
        assert "intake" in out

    def test_custom_dir(self, capsys, monkeypatch, tmp_path):
        (tmp_path / "custom.txt").write_text("Custom template for $transcript")
        monkeypatch.setattr("sys.argv", ["mn-templates", "--dir", str(tmp_path)])
        cli.templates()
        out = capsys.readouterr().out
        assert "custom" in out

    def test_missing_dir_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mn-templates", "--dir", "/nonexistent"])
        with pytest.raises(SystemExit):
            cli.templates()


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
        assert "[00:00" in out

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

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=_sample_segments()):
            with patch("mn.summarize.httpx.post",
                        return_value=_mock_llm_response("Final note")):
                cli.main()

        out = capsys.readouterr().out
        assert "Final note" in out

    def test_speakers_flag(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn", "test.wav",
            "--template", str(template),
            "--transcript-only",
            "--speakers", "Therapist,Client",
        ])

        with patch("mn.transcribe.transcribe_and_diarize",
                    return_value=_sample_segments()):
            cli.main()

        out = capsys.readouterr().out
        assert "Therapist:" in out
        assert "Client:" in out

    def test_redact_flag(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        segs = [Segment("A", "SSN is 123-45-6789.", 0.0, 1.0)]
        monkeypatch.setattr("sys.argv", [
            "mn", "test.wav",
            "--template", str(template),
            "--transcript-only",
            "--redact",
        ])

        with patch("mn.transcribe.transcribe_and_diarize", return_value=segs):
            cli.main()

        out = capsys.readouterr().out
        assert "[SSN]" in out
        assert "123-45-6789" not in out

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
                _whisper=None,
                _diarizer=None,
            )


# -- mn-batch ---------------------------------------------------------------


class TestBatchCli:

    def test_processes_directory(self, capsys, monkeypatch, tmp_path):
        # Create a fake audio file.
        (tmp_path / "session1.wav").write_bytes(b"fake audio")
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
            "--transcript-only",
        ])

        with patch("mn.transcribe.load_whisper", return_value="whisper_mock"):
            with patch("mn.transcribe.load_diarizer", return_value="diarizer_mock"):
                with patch("mn.transcribe.transcribe_and_diarize",
                            return_value=_sample_segments()):
                    cli.batch()

        out_file = tmp_path / "session1.txt"
        assert out_file.exists()
        content = out_file.read_text()
        assert "SPEAKER_00" in content

    def test_models_loaded_once_for_batch(self, capsys, monkeypatch, tmp_path):
        """Whisper and diarizer should be loaded once, not per file."""
        for name in ["a.wav", "b.wav", "c.wav"]:
            (tmp_path / name).write_bytes(b"fake")
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
            "--transcript-only",
        ])

        with patch("mn.transcribe.load_whisper",
                    return_value="w") as whisper_load:
            with patch("mn.transcribe.load_diarizer",
                        return_value="d") as diarizer_load:
                with patch("mn.transcribe.transcribe_and_diarize",
                            return_value=_sample_segments()) as tad:
                    cli.batch()

        # Models loaded exactly once, not 3 times.
        whisper_load.assert_called_once()
        diarizer_load.assert_called_once()
        # But transcribe_and_diarize called 3 times (once per file).
        assert tad.call_count == 3
        # Each call should pass through the pre-loaded models.
        for call in tad.call_args_list:
            assert call[1]["_whisper"] == "w"
            assert call[1]["_diarizer"] == "d"

    def test_output_dir(self, capsys, monkeypatch, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "s.wav").write_bytes(b"fake")
        out_dir = tmp_path / "notes"

        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(audio_dir),
            "--template", str(template),
            "--output-dir", str(out_dir),
            "--transcript-only",
        ])

        with patch("mn.transcribe.load_whisper", return_value="w"):
            with patch("mn.transcribe.load_diarizer", return_value="d"):
                with patch("mn.transcribe.transcribe_and_diarize",
                            return_value=_sample_segments()):
                    cli.batch()

        assert (out_dir / "s.txt").exists()

    def test_no_files_exits(self, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
        ])

        with pytest.raises(SystemExit):
            cli.batch()
