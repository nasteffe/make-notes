"""Summarize a transcript using a plain-text template and an LLM.

Templates are text files with $-placeholders (string.Template syntax):

    $transcript    — the full formatted transcript
    $speakers      — comma-separated list of speakers
    $date          — session date (default: today)
    $duration      — session duration derived from timestamps
    $client_name   — client name if provided

This talks to any OpenAI-compatible chat completions endpoint,
so it works with ollama, vllm, together, openai, lmstudio, etc.

Configure via environment or CLI flags:

    MN_API_BASE  (default: http://localhost:11434/v1  — ollama)
    MN_MODEL     (default: llama3)
    MN_API_KEY   (default: ollama)
"""

import json
import os
import sys
from datetime import date
from pathlib import Path
from string import Template

from urllib.parse import urlparse

import httpx

# Rough chars-per-token ratio for English text. Used for warnings only.
_CHARS_PER_TOKEN = 4
_TOKEN_WARNING_THRESHOLD = 6000

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

from .fmt import fmt
from .transcribe import Segment


def load_template(path):
    """Read a template file from disk."""
    return Path(path).read_text()


def _duration(segments):
    """Compute human-readable duration from segment timestamps."""
    if not segments:
        return "0:00"
    start = min(s.start for s in segments)
    end = max(s.end for s in segments)
    total = int(end - start)
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def render(template_text, segments, client_name=None, session_date=None):
    """Fill $placeholders in a template with transcript data."""
    transcript = fmt(segments, timestamps=True)
    speakers = ", ".join(sorted(set(s.speaker for s in segments)))
    return Template(template_text).safe_substitute(
        transcript=transcript,
        speakers=speakers,
        date=session_date or date.today().isoformat(),
        duration=_duration(segments),
        client_name=client_name or "Client",
    )


def _estimate_tokens(text):
    """Rough token estimate: len(text) / 4. For warnings only."""
    return len(text) // _CHARS_PER_TOKEN


def _is_local(url):
    """Return True if the URL points to a local/loopback address."""
    try:
        host = urlparse(url).hostname or ""
        return host in _LOCAL_HOSTS
    except Exception:
        return False


class RemoteEndpointError(RuntimeError):
    """Raised when a remote LLM endpoint is used without --allow-remote."""


def complete(prompt, base_url=None, model=None, api_key=None,
             allow_remote=False):
    """Send a prompt to an OpenAI-compatible chat completions endpoint.

    Raises RemoteEndpointError if the endpoint is non-local and
    allow_remote is False.
    """
    base_url = base_url or os.environ.get("MN_API_BASE",
                                           "http://localhost:11434/v1")
    model = model or os.environ.get("MN_MODEL", "llama3")
    api_key = api_key or os.environ.get("MN_API_KEY", "ollama")

    if not _is_local(base_url):
        if not allow_remote:
            raise RemoteEndpointError(
                f"Refusing to send clinical data to remote endpoint "
                f"({base_url}). Pass --allow-remote to confirm, or use "
                f"a local LLM (e.g. ollama)."
            )
        print(
            f"Warning: sending transcript to remote endpoint ({base_url}). "
            f"Ensure you have appropriate data handling agreements in place.",
            file=sys.stderr,
        )

    est = _estimate_tokens(prompt)
    if est > _TOKEN_WARNING_THRESHOLD:
        print(
            f"Warning: prompt is ~{est} tokens. Models with small context "
            f"windows may truncate or fail. Consider using a larger model "
            f"or splitting the session.",
            file=sys.stderr,
        )

    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300.0,
    )
    resp.raise_for_status()
    body = resp.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected LLM response format: {json.dumps(body)[:200]}"
        )


def summarize(segments, template_path, client_name=None, session_date=None,
              **llm_kwargs):
    """Segments + template file → summary text from LLM."""
    template_text = load_template(template_path)
    prompt = render(template_text, segments, client_name=client_name,
                    session_date=session_date)
    return complete(prompt, **llm_kwargs)
