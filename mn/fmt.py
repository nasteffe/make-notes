"""Format diarized transcript segments as readable text.

Reads Segments, writes plain text. The universal interchange format.

    SPEAKER_00: I've been feeling anxious this week.

    SPEAKER_01: Can you tell me more about that?
"""

from .transcribe import Segment


def fmt(segments, timestamps=False):
    """Segments → readable multi-party transcript string."""
    return "\n\n".join(_format_line(s, timestamps) for s in segments)


def _format_line(seg, timestamps):
    if timestamps:
        return f"[{_ftime(seg.start)} → {_ftime(seg.end)}] {seg.speaker}: {seg.text}"
    return f"{seg.speaker}: {seg.text}"


def _ftime(seconds):
    """Seconds → MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
