"""Transcribe audio with speaker diarization.

Pipeline: audio → words → speaker segments → aligned transcript.

Each stage is a pure function. Compose them, or use transcribe_and_diarize
for the full pipeline. Output is JSON lines — one segment per line:

    {"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 2.5}
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Segment:
    """A span of speech attributed to one speaker."""
    speaker: str
    text: str
    start: float
    end: float


# -- Stage 1: Transcribe ------------------------------------------------


def load_whisper(model_size="base", device="cpu", compute_type="int8"):
    """Load a WhisperModel. Reuse the returned object to avoid reloading."""
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(audio_path, model_size="base", device="cpu", compute_type="int8",
               _model=None):
    """Audio file → list of word dicts with timestamps.

    Pass _model (a WhisperModel) to skip loading — useful for batch mode.
    """
    if _model is None:
        _model = load_whisper(model_size, device, compute_type)

    segments, _info = _model.transcribe(str(audio_path), word_timestamps=True)

    words = []
    for seg in segments:
        if seg.words is None:
            continue
        for w in seg.words:
            words.append({"text": w.word, "start": w.start, "end": w.end})
    return words


# -- Stage 2: Diarize ---------------------------------------------------


def load_diarizer(hf_token=None):
    """Load the pyannote diarization pipeline. Reuse to avoid reloading."""
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )


def diarize(audio_path, num_speakers=None, min_speakers=None,
            max_speakers=None, hf_token=None, _pipeline=None):
    """Audio file → list of speaker segment dicts.

    Pass _pipeline (a pyannote Pipeline) to skip loading — useful for batch.
    """
    if _pipeline is None:
        _pipeline = load_diarizer(hf_token)

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    result = _pipeline(str(audio_path), **params)

    return [
        {"speaker": speaker, "start": turn.start, "end": turn.end}
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]


# -- Stage 3: Align -----------------------------------------------------


def align(words, speaker_segments):
    """Merge word timestamps with speaker segments → list of Segments.

    Each word is assigned to the speaker whose segment overlaps it most.
    Consecutive words from the same speaker are merged into one Segment.
    """
    if not words:
        return []

    if not speaker_segments:
        # No diarization — attribute everything to a single speaker.
        text = "".join(w["text"] for w in words).strip()
        return [Segment("Speaker", text, words[0]["start"], words[-1]["end"])]

    def _find_speaker(start, end):
        mid = (start + end) / 2
        best, best_overlap = "Unknown", 0.0
        for seg in speaker_segments:
            ov = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
            if ov > best_overlap:
                best, best_overlap = seg["speaker"], ov
        if best_overlap == 0.0:
            for seg in speaker_segments:
                if seg["start"] <= mid <= seg["end"]:
                    return seg["speaker"]
        return best

    # Attribute each word, then merge runs of the same speaker.
    attributed = [
        {**w, "speaker": _find_speaker(w["start"], w["end"])}
        for w in words
    ]

    segments = []
    cur = None
    for w in attributed:
        if cur is None or w["speaker"] != cur["speaker"]:
            if cur:
                cur["text"] = cur["text"].strip()
                segments.append(Segment(**cur))
            cur = dict(speaker=w["speaker"], text=w["text"],
                       start=w["start"], end=w["end"])
        else:
            cur["text"] += w["text"]
            cur["end"] = w["end"]

    if cur:
        cur["text"] = cur["text"].strip()
        segments.append(Segment(**cur))

    return segments


# -- Composed pipeline ---------------------------------------------------


def transcribe_and_diarize(audio_path, model_size="base", device="cpu",
                           compute_type="int8", num_speakers=None,
                           min_speakers=None, max_speakers=None,
                           hf_token=None, _whisper=None, _diarizer=None):
    """Full pipeline: audio file → list of diarized Segments.

    Pass _whisper and _diarizer to reuse pre-loaded models (batch mode).
    """
    words = transcribe(audio_path, model_size, device, compute_type,
                       _model=_whisper)
    speakers = diarize(audio_path, num_speakers, min_speakers,
                       max_speakers, hf_token, _pipeline=_diarizer)
    return align(words, speakers)


# -- Speaker labeling ----------------------------------------------------


def label_speakers(segments, names):
    """Replace generic SPEAKER_XX labels with human names.

    names is a comma-separated string or list: "Therapist,Client"
    Speakers are assigned in order of first appearance.
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]

    # Map each unique speaker label → name, in order of first appearance.
    seen = []
    for s in segments:
        if s.speaker not in seen:
            seen.append(s.speaker)

    mapping = {}
    for i, original in enumerate(seen):
        mapping[original] = names[i] if i < len(names) else original

    return [
        Segment(mapping.get(s.speaker, s.speaker), s.text, s.start, s.end)
        for s in segments
    ]


# -- Serialization -------------------------------------------------------


def to_jsonl(segments):
    """Segments → JSON lines string."""
    return "\n".join(json.dumps(asdict(s)) for s in segments)


def from_jsonl(text):
    """JSON lines string → list of Segments."""
    return [
        Segment(**json.loads(line))
        for line in text.strip().split("\n")
        if line.strip()
    ]
