"""Tests for mn.summarize — template rendering and LLM integration."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mn.summarize import complete, load_template, render, summarize
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
            result = complete("prompt", base_url="http://test/v1",
                              model="m", api_key="k")
        assert result == "OK"

    def test_uses_env_defaults(self, monkeypatch):
        monkeypatch.setenv("MN_API_BASE", "http://env-base/v1")
        monkeypatch.setenv("MN_MODEL", "env-model")
        monkeypatch.setenv("MN_API_KEY", "env-key")

        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("test prompt")
            call_kwargs = mock.call_args
            assert "env-base" in call_kwargs[0][0]
            body = call_kwargs[1]["json"]
            assert body["model"] == "env-model"
            assert "env-key" in call_kwargs[1]["headers"]["Authorization"]

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("MN_API_BASE", "http://env/v1")
        monkeypatch.setenv("MN_MODEL", "env-model")

        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://explicit/v1", model="explicit-model")
            call_kwargs = mock.call_args
            assert "explicit" in call_kwargs[0][0]
            assert call_kwargs[1]["json"]["model"] == "explicit-model"

    def test_strips_trailing_slash_from_base_url(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://test/v1/", model="m", api_key="k")
            url = mock.call_args[0][0]
            assert "/v1/chat/completions" in url
            assert "//chat" not in url

    def test_http_error_raises(self):
        error_resp = httpx.Response(
            500,
            json={"error": "server error"},
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )
        with patch("mn.summarize.httpx.post", return_value=error_resp):
            with pytest.raises(httpx.HTTPStatusError):
                complete("prompt", base_url="http://test/v1",
                         model="m", api_key="k")

    def test_temperature_is_0_3(self):
        with patch("mn.summarize.httpx.post", return_value=_mock_response()) as mock:
            complete("prompt", base_url="http://test/v1", model="m", api_key="k")
            body = mock.call_args[1]["json"]
            assert body["temperature"] == 0.3


# -- summarize() end-to-end with mock LLM ----------------------------------


class TestSummarize:

    def test_end_to_end(self, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Summarize: $transcript")

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_response("SOAP note here")):
            result = summarize(_segments(), str(template),
                               base_url="http://test/v1",
                               model="m", api_key="k")
        assert result == "SOAP note here"

    def test_template_content_sent_to_llm(self, tmp_path):
        template = tmp_path / "t.txt"
        template.write_text("Template: $speakers\n$transcript")

        with patch("mn.summarize.httpx.post",
                    return_value=_mock_response("result")) as mock:
            summarize(_segments(), str(template),
                      base_url="http://test/v1", model="m", api_key="k")
            prompt = mock.call_args[1]["json"]["messages"][0]["content"]
            assert "SPEAKER_00" in prompt
            assert "SPEAKER_01" in prompt
            assert "Template:" in prompt


# -- Template files in templates/ -------------------------------------------


class TestTemplateFiles:
    """Verify that the shipped templates are valid."""

    @pytest.fixture(params=["soap.txt", "dap.txt", "birp.txt", "progress.txt"])
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

    def test_template_renders_without_error(self, template_path):
        text = template_path.read_text()
        result = render(text, _segments())
        assert "SPEAKER_00" in result
        assert len(result) > len(text)  # transcript was substituted
