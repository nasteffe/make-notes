"""Open a transcript in $EDITOR for manual correction.

Round-trips through a human-readable format:

    [00:00 → 00:05] SPEAKER_00:
    I've been feeling anxious this week.

    [00:05 → 00:10] SPEAKER_01:
    Can you tell me more about that?

Edit speaker labels, fix transcription errors, then save and quit.
The corrected segments are written as JSON lines to stdout.
"""

import os
import re
import subprocess
import sys
import tempfile

from .fmt import _ftime
from .transcribe import Segment


# -- Serialize to editable format -------------------------------------------


def to_editable(segments):
    """Segments → human-editable text (round-trippable)."""
    blocks = []
    for s in segments:
        header = f"[{_ftime(s.start)} → {_ftime(s.end)}] {s.speaker}:"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n".join(blocks) + "\n"


# -- Parse back from editable format ----------------------------------------

_HEADER = re.compile(
    r"^\[(\d+:\d+)\s*→\s*(\d+:\d+)\]\s*(.+?)\s*:$"
)


def _parse_time(ts):
    """MM:SS → float seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def from_editable(text):
    """Human-editable text → list of Segments."""
    segments = []
    blocks = re.split(r"\n\n+", text.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        m = _HEADER.match(lines[0])
        if not m:
            continue

        start = float(_parse_time(m.group(1)))
        end = float(_parse_time(m.group(2)))
        speaker = m.group(3).strip()
        body = "\n".join(lines[1:]).strip()
        segments.append(Segment(speaker, body, start, end))

    return segments


# -- Editor launcher --------------------------------------------------------


def edit(segments):
    """Open segments in $EDITOR, return corrected segments."""
    editor = os.environ.get("EDITOR", "vi")
    text = to_editable(segments)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="mn-edit-", delete=False
    ) as f:
        f.write(text)
        tmp = f.name

    try:
        subprocess.run([editor, tmp], check=True)
        with open(tmp) as f:
            edited = f.read()
    finally:
        os.unlink(tmp)

    return from_editable(edited)
