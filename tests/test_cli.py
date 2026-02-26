"""Tests for mn.cli — argument parsing and pipeline wiring."""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mn import cli
from mn.cli import _check_audio_file, _check_hf_token, _check_template, _die
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
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
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
            "--base-url", "http://localhost:11434/v1",
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
            "--base-url", "http://localhost:11434/v1",
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
            "--base-url", "http://localhost:11434/v1",
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

    @pytest.fixture(autouse=True)
    def _skip_validation(self, monkeypatch):
        """Skip file/token validation in transcribe CLI tests."""
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        with patch("mn.cli._check_audio_file"):
            yield

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
        ])
        monkeypatch.setenv("HF_TOKEN", "hf_test")

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
        assert "neuropsychoanalytic" in out
        assert "informed-consent" in out

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

    @pytest.fixture(autouse=True)
    def _skip_validation(self, monkeypatch):
        """Skip file/token validation in main CLI tests."""
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        with patch("mn.cli._check_audio_file"):
            yield

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
            "--base-url", "http://localhost:11434/v1",
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
                _whisper=None,
                _diarizer=None,
            )


# -- mn-batch ---------------------------------------------------------------


class TestBatchCli:

    @pytest.fixture(autouse=True)
    def _skip_validation(self, monkeypatch):
        """Skip file/token validation in batch CLI tests."""
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        with patch("mn.cli._check_audio_file"):
            yield

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

    def test_mid_batch_failure_continues(self, capsys, monkeypatch, tmp_path):
        """When one file fails transcription, remaining files still process."""
        for name in ["a.wav", "b.wav", "c.wav"]:
            (tmp_path / name).write_bytes(b"fake")
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
            "--transcript-only", "-vv",
        ])

        call_count = 0

        def flaky_transcribe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Second file fails.
                raise RuntimeError("Transcription failed")
            return _sample_segments()

        with patch("mn.transcribe.load_whisper", return_value="w"):
            with patch("mn.transcribe.load_diarizer", return_value="d"):
                with patch("mn.transcribe.transcribe_and_diarize",
                            side_effect=flaky_transcribe):
                    cli.batch()

        # a.txt and c.txt should exist; b.txt should not.
        assert (tmp_path / "a.txt").exists()
        assert not (tmp_path / "b.txt").exists()
        assert (tmp_path / "c.txt").exists()

        # Progress output should mention the failure.
        err = capsys.readouterr().err
        assert "1 file(s) failed" in err
        assert "b.wav" in err

    def test_all_files_fail(self, capsys, monkeypatch, tmp_path):
        """When all files fail, batch still completes with a summary."""
        (tmp_path / "x.wav").write_bytes(b"fake")
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
            "--transcript-only", "-vv",
        ])

        with patch("mn.transcribe.load_whisper", return_value="w"):
            with patch("mn.transcribe.load_diarizer", return_value="d"):
                with patch("mn.transcribe.transcribe_and_diarize",
                            side_effect=RuntimeError("fail")):
                    cli.batch()

        assert not (tmp_path / "x.txt").exists()
        err = capsys.readouterr().err
        assert "1 file(s) failed" in err
        assert "0/1" in err

    def test_summarize_failure_continues(self, capsys, monkeypatch, tmp_path):
        """When LLM summarization fails for one file, others still process."""
        for name in ["a.wav", "b.wav", "c.wav"]:
            (tmp_path / name).write_bytes(b"fake")
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.argv", [
            "mn-batch", str(tmp_path),
            "--template", str(template),
            "--base-url", "http://localhost:11434/v1",
            "--llm-model", "m", "--api-key", "k",
            "-vv",
        ])

        call_count = 0

        def flaky_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ConnectError("LLM is down")
            return _mock_llm_response("Generated note")

        with patch("mn.transcribe.load_whisper", return_value="w"):
            with patch("mn.transcribe.load_diarizer", return_value="d"):
                with patch("mn.transcribe.transcribe_and_diarize",
                            return_value=_sample_segments()):
                    with patch("mn.summarize.httpx.post",
                                side_effect=flaky_llm):
                        cli.batch()

        # a and c should have notes; b should not.
        assert (tmp_path / "a.note.txt").exists()
        assert not (tmp_path / "b.note.txt").exists()
        assert (tmp_path / "c.note.txt").exists()

        err = capsys.readouterr().err
        assert "1 file(s) failed" in err
        assert "b.wav" in err


# -- Error handling helpers -------------------------------------------------


class TestDie:

    def test_prints_to_stderr_and_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _die("something went wrong")
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Error: something went wrong" in err


class TestCheckAudioFile:

    def test_passes_for_valid_file(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)
        _check_audio_file(str(f))  # should not raise

    def test_fails_for_missing_file(self):
        with pytest.raises(SystemExit):
            _check_audio_file("/nonexistent/audio.wav")

    def test_fails_for_directory(self, tmp_path):
        with pytest.raises(SystemExit):
            _check_audio_file(str(tmp_path))

    def test_fails_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        with pytest.raises(SystemExit):
            _check_audio_file(str(f))


class TestCheckHfToken:

    def test_passes_with_env_var(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        _check_hf_token()  # should not raise

    def test_fails_without_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            _check_hf_token()


class TestCheckTemplate:

    def test_passes_for_valid_file(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("$transcript")
        _check_template(str(f))  # should not raise

    def test_fails_for_missing_file(self):
        with pytest.raises(SystemExit):
            _check_template("/nonexistent/template.txt")

    def test_passes_for_none(self):
        _check_template(None)  # should not raise


# -- Progress and config integration ----------------------------------------


class TestProgressReporting:

    def test_transcribe_prints_progress(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", "test.wav", "-vv",
        ])
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        with patch("mn.cli._check_audio_file"):
            with patch("mn.transcribe.transcribe_and_diarize",
                        return_value=_sample_segments()):
                cli.transcribe()

        err = capsys.readouterr().err
        assert "Transcribing" in err
        assert "2 segments" in err

    def test_summarize_prints_progress(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--base-url", "http://localhost:11434/v1",
            "--llm-model", "m", "--api-key", "k",
            "-vv",
        ])

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_llm_response()):
            cli.summarize()

        err = capsys.readouterr().err
        assert "Summarizing" in err

    def test_quiet_mode_suppresses_progress(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", "test.wav",
        ])
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        monkeypatch.setenv("MN_VERBOSE", "0")

        with patch("mn.cli._check_audio_file"):
            with patch("mn.transcribe.transcribe_and_diarize",
                        return_value=_sample_segments()):
                cli.transcribe()

        err = capsys.readouterr().err
        assert "Transcribing" not in err


class TestConfigIntegration:

    def test_config_fills_defaults(self, capsys, monkeypatch, tmp_path):
        # Write a config file.
        cfg = tmp_path / "mn.toml"
        cfg.write_text(
            "[transcribe]\n"
            'speakers = "Therapist,Client"\n'
        )
        monkeypatch.chdir(tmp_path)

        # Create a fake audio file.
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")

        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", str(audio),
        ])
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        with patch("mn.cli._check_audio_file"):
            with patch("mn.transcribe.transcribe_and_diarize",
                        return_value=_sample_segments()):
                cli.transcribe()

        out = capsys.readouterr().out
        lines = [json.loads(l) for l in out.strip().split("\n")]
        # Config set speakers=Therapist,Client, so labels should be applied.
        assert lines[0]["speaker"] == "Therapist"
        assert lines[1]["speaker"] == "Client"


# -- Remote endpoint gate in CLI -------------------------------------------


class TestRemoteEndpointCli:

    def test_summarize_blocks_remote_without_flag(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--base-url", "https://api.openai.com/v1",
            "--llm-model", "m", "--api-key", "k",
        ])

        with pytest.raises(SystemExit):
            cli.summarize()
        err = capsys.readouterr().err
        assert "Refusing" in err or "remote" in err.lower()

    def test_summarize_allows_remote_with_flag(self, capsys, monkeypatch, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("$transcript")

        monkeypatch.setattr("sys.stdin", StringIO(_sample_jsonl()))
        monkeypatch.setattr("sys.argv", [
            "mn-summarize", "--template", str(template),
            "--base-url", "https://api.openai.com/v1",
            "--llm-model", "m", "--api-key", "k",
            "--allow-remote",
        ])

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_llm_response()):
            cli.summarize()

        out = capsys.readouterr().out
        assert "Generated note" in out


# -- Malformed config file -------------------------------------------------


class TestMalformedConfig:

    def test_malformed_toml_warns_and_continues(self, capsys, monkeypatch, tmp_path):
        cfg = tmp_path / "mn.toml"
        cfg.write_text("this is not valid [[ toml")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        monkeypatch.setattr("sys.argv", [
            "mn-transcribe", str(tmp_path / "test.wav"),
        ])

        with patch("mn.cli._check_audio_file"):
            with patch("mn.transcribe.transcribe_and_diarize",
                        return_value=_sample_segments()):
                cli.transcribe()

        err = capsys.readouterr().err
        assert "could not parse" in err.lower()
