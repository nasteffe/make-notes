"""Tests for mn.summarize — template rendering and LLM integration."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mn.summarize import (
    RemoteEndpointError,
    _TOKEN_WARNING_THRESHOLD,
    _duration,
    _estimate_tokens,
    _is_local,
    complete,
    load_template,
    render,
    summarize,
)
from mn.transcribe import Segment


def _segments():
    return [
        Segment("SPEAKER_00", "I've been anxious.", 0.0, 1.2),
        Segment("SPEAKER_01", "Tell me more.", 1.5, 2.6),
    ]


# -- load_template() --------------------------------------------------------


class TestLoadTemplate:

    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello $transcript")
        assert load_template(f) == "Hello $transcript"

    def test_preserves_newlines(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("line1\nline2\nline3")
        assert load_template(f).count("\n") == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_template(tmp_path / "nope.txt")


# -- render() ---------------------------------------------------------------


class TestRender:

    def test_substitutes_transcript(self):
        result = render("BEGIN\n$transcript\nEND", _segments())
        assert "SPEAKER_00:" in result
        assert "SPEAKER_01:" in result
        assert "BEGIN" in result
        assert "END" in result

    def test_substitutes_speakers(self):
        result = render("Speakers: $speakers", _segments())
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result

    def test_speakers_are_sorted(self):
        segs = [
            Segment("Zara", "hi", 0.0, 1.0),
            Segment("Alice", "hey", 1.0, 2.0),
        ]
        result = render("$speakers", segs)
        assert result.strip() == "Alice, Zara"

    def test_speakers_are_deduplicated(self):
        segs = [
            Segment("A", "one", 0.0, 1.0),
            Segment("B", "two", 1.0, 2.0),
            Segment("A", "three", 2.0, 3.0),
        ]
        result = render("$speakers", segs)
        assert result.strip() == "A, B"

    def test_unknown_placeholders_left_alone(self):
        result = render("$transcript $unknown_thing", _segments())
        assert "$unknown_thing" in result

    def test_transcript_includes_timestamps(self):
        result = render("$transcript", _segments())
        assert "[00:00" in result
        assert "→" in result

    def test_curly_braces_in_template_preserved(self):
        result = render("json: {\"key\": \"val\"}\n$transcript", _segments())
        assert '{"key": "val"}' in result

    def test_dollar_literal_preserved(self):
        result = render("costs $$5\n$transcript", _segments())
        assert "$5" in result

    def test_substitutes_date_default(self):
        from datetime import date
        result = render("Date: $date", _segments())
        assert date.today().isoformat() in result

    def test_substitutes_date_explicit(self):
        result = render("Date: $date", _segments(), session_date="2026-01-15")
        assert "2026-01-15" in result

    def test_substitutes_duration(self):
        segs = [
            Segment("A", "start", 0.0, 30.0),
            Segment("B", "end", 30.0, 125.0),
        ]
        result = render("Duration: $duration", segs)
        assert "2:05" in result

    def test_substitutes_client_name_default(self):
        result = render("Client: $client_name", _segments())
        assert "Client" in result

    def test_substitutes_client_name_explicit(self):
        result = render("Client: $client_name", _segments(), client_name="J.D.")
        assert "J.D." in result


# -- _duration() ------------------------------------------------------------


class TestDuration:

    def test_basic(self):
        segs = [Segment("A", "x", 0.0, 60.0)]
        assert _duration(segs) == "1:00"

    def test_multiple_segments(self):
        segs = [
            Segment("A", "x", 10.0, 30.0),
            Segment("B", "y", 30.0, 135.0),
        ]
        assert _duration(segs) == "2:05"

    def test_empty(self):
        assert _duration([]) == "0:00"

    def test_zero_duration(self):
        segs = [Segment("A", "x", 5.0, 5.0)]
        assert _duration(segs) == "0:00"


# -- complete() with mocked HTTP -------------------------------------------


def _mock_response(content="Summary text"):
    """Build a mock httpx.Response for chat completions."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
        },
        request=httpx.Request("POST", "http://test/v1/chat/completions"),
    )


class TestComplete:

    def test_returns_message_content(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response("OK")):
            result = complete("prompt", base_url="http://localhost:11434/v1",
                              model="m", api_key="k")
        assert result == "OK"

    def test_uses_env_defaults(self, monkeypatch):
        monkeypatch.setenv("MN_API_BASE", "http://localhost:9999/v1")
        monkeypatch.setenv("MN_MODEL", "env-model")
        monkeypatch.setenv("MN_API_KEY", "env-key")

        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("test prompt")
            call_kwargs = mock.call_args
            assert "localhost:9999" in call_kwargs[0][0]
            body = call_kwargs[1]["json"]
            assert body["model"] == "env-model"
            assert "env-key" in call_kwargs[1]["headers"]["Authorization"]

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("MN_API_BASE", "http://localhost:1111/v1")
        monkeypatch.setenv("MN_MODEL", "env-model")

        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://localhost:2222/v1",
                     model="explicit-model")
            call_kwargs = mock.call_args
            assert "localhost:2222" in call_kwargs[0][0]
            assert call_kwargs[1]["json"]["model"] == "explicit-model"

    def test_strips_trailing_slash_from_base_url(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://localhost:11434/v1/", model="m", api_key="k")
            url = mock.call_args[0][0]
            assert "/v1/chat/completions" in url
            assert "//chat" not in url

    def test_http_error_raises(self):
        error_resp = httpx.Response(
            500,
            json={"error": "server error"},
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=error_resp):
            with pytest.raises(httpx.HTTPStatusError):
                complete("prompt", base_url="http://localhost:11434/v1",
                         model="m", api_key="k")

    def test_temperature_is_0_3(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://localhost:11434/v1", model="m", api_key="k")
            body = mock.call_args[1]["json"]
            assert body["temperature"] == 0.3

    def test_timeout_is_300(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://localhost:11434/v1", model="m", api_key="k")
            assert mock.call_args[1]["timeout"] == 300.0

    def test_warns_on_long_prompt(self, capsys):
        # Generate a prompt that exceeds the token warning threshold.
        long_prompt = "x" * (_TOKEN_WARNING_THRESHOLD * 4 + 100)
        with patch("mn.summarize.httpx.post", return_value=_mock_response()):
            complete(long_prompt, base_url="http://localhost:11434/v1",
                     model="m", api_key="k")
        err = capsys.readouterr().err
        assert "Warning" in err
        assert "tokens" in err

    def test_no_warning_on_short_local_prompt(self, capsys):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()):
            complete("short prompt", base_url="http://localhost:11434/v1",
                     model="m", api_key="k")
        err = capsys.readouterr().err
        assert err == ""


# -- _estimate_tokens() ----------------------------------------------------


class TestEstimateTokens:

    def test_empty(self):
        assert _estimate_tokens("") == 0

    def test_basic(self):
        # 20 chars / 4 = 5 tokens
        assert _estimate_tokens("a" * 20) == 5


# -- _is_local() -----------------------------------------------------------


class TestIsLocal:

    def test_localhost(self):
        assert _is_local("http://localhost:11434/v1") is True

    def test_127_0_0_1(self):
        assert _is_local("http://127.0.0.1:11434/v1") is True

    def test_0_0_0_0(self):
        assert _is_local("http://0.0.0.0:8000/v1") is True

    def test_ipv6_loopback(self):
        assert _is_local("http://[::1]:11434/v1") is True

    def test_remote_url(self):
        assert _is_local("https://api.openai.com/v1") is False

    def test_custom_domain(self):
        assert _is_local("https://my-server.example.com/v1") is False

    def test_invalid_url(self):
        assert _is_local("not a url") is False


# -- Cloud PII warning ----------------------------------------------------


class TestRemoteEndpointGate:

    def test_blocks_remote_without_allow_remote(self):
        with pytest.raises(RemoteEndpointError, match="Refusing"):
            complete("prompt", base_url="https://api.openai.com/v1",
                     model="m", api_key="k")

    def test_warns_on_remote_with_allow_remote(self, capsys):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()):
            complete("prompt", base_url="https://api.openai.com/v1",
                     model="m", api_key="k", allow_remote=True)
        err = capsys.readouterr().err
        assert "remote endpoint" in err
        assert "data handling" in err

    def test_no_warning_on_localhost(self, capsys):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()):
            complete("prompt", base_url="http://localhost:11434/v1",
                     model="m", api_key="k")
        err = capsys.readouterr().err
        assert "remote endpoint" not in err

    def test_no_warning_on_127_0_0_1(self, capsys):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()):
            complete("prompt", base_url="http://127.0.0.1:11434/v1",
                     model="m", api_key="k")
        err = capsys.readouterr().err
        assert "remote endpoint" not in err


# -- summarize() end-to-end with mock LLM ----------------------------------


class TestResponseValidation:

    def test_empty_choices_raises(self):
        bad_resp = httpx.Response(
            200,
            json={"choices": []},
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="Unexpected LLM response"):
                complete("prompt", base_url="http://localhost:11434/v1",
                         model="m", api_key="k")

    def test_missing_message_key_raises(self):
        bad_resp = httpx.Response(
            200,
            json={"choices": [{"not_message": "x"}]},
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="Unexpected LLM response"):
                complete("prompt", base_url="http://localhost:11434/v1",
                         model="m", api_key="k")

    def test_no_choices_key_raises(self):
        bad_resp = httpx.Response(
            200,
            json={"error": "something"},
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="Unexpected LLM response"):
                complete("prompt", base_url="http://localhost:11434/v1",
                         model="m", api_key="k")


# -- summarize() end-to-end with mock LLM ----------------------------------


class TestSummarize:

    def test_end_to_end(self, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Summarize: $transcript")

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_response("SOAP note here")):
            result = summarize(_segments(), str(template),
                               base_url="http://localhost:11434/v1",
                               model="m", api_key="k")
        assert result == "SOAP note here"

    def test_template_content_sent_to_llm(self, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Template: $speakers\n$transcript")

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_response("result")) as mock:
            summarize(_segments(), str(template),
                      base_url="http://localhost:11434/v1", model="m", api_key="k")
            prompt = mock.call_args[1]["json"]["messages"][0]["content"]
            assert "SPEAKER_00" in prompt
            assert "SPEAKER_01" in prompt
            assert "Template:" in prompt


# -- Template files in templates/ -------------------------------------------


class TestTemplateFiles:
    """Verify that the shipped templates are valid."""

    @pytest.fixture(params=[
        "soap.txt", "dap.txt", "birp.txt", "progress.txt",
        "cbt-soap.txt", "psychodynamic.txt", "intake.txt",
    ])
    def template_path(self, request):
        p = Path(__file__).parent.parent / "templates" / request.param
        assert p.exists(), f"Template missing: {p}"
        return p

    def test_template_contains_transcript_placeholder(self, template_path):
        text = template_path.read_text()
        assert "$transcript" in text

    def test_template_contains_speakers_placeholder(self, template_path):
        text = template_path.read_text()
        assert "$speakers" in text

    def test_template_contains_client_name_placeholder(self, template_path):
        text = template_path.read_text()
        assert "$client_name" in text

    def test_template_contains_date_placeholder(self, template_path):
        text = template_path.read_text()
        assert "$date" in text

    def test_template_contains_draft_disclaimer(self, template_path):
        text = template_path.read_text()
        assert "draft" in text.lower() or "clinician review" in text.lower()

    def test_template_renders_without_error(self, template_path):
        text = template_path.read_text()
        result = render(text, _segments())
        assert "SPEAKER_00" in result
        assert len(result) > len(text)  # transcript was substituted
