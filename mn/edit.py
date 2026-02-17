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
    """Human-editable text → list of Segments.

    Splits on header lines rather than blank lines, so body text can
    safely contain blank lines without breaking the parse.
    """
    segments = []
    lines = text.split("\n")

    # Collect (header_match, start_line_index) pairs.
    headers = []
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if m:
            headers.append((m, i))

    for idx, (m, start_i) in enumerate(headers):
        # Body extends from line after header to line before next header.
        if idx + 1 < len(headers):
            end_i = headers[idx + 1][1]
        else:
            end_i = len(lines)

        body = "\n".join(lines[start_i + 1:end_i]).strip()
        segments.append(Segment(
            m.group(3).strip(),
            body,
            float(_parse_time(m.group(1))),
            float(_parse_time(m.group(2))),
        ))

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
