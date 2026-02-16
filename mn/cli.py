"""CLI entry points for make-notes.

Four tools, each useful alone, composable together:

    mn-transcribe session.wav > transcript.jsonl
    mn-fmt < transcript.jsonl
    mn-fmt --timestamps < transcript.jsonl
    mn-summarize --template templates/soap.txt < transcript.jsonl
    mn session.wav --template templates/soap.txt

Stdin/stdout everywhere. JSON lines as the interchange format.
"""

import argparse
import sys

from . import fmt as _fmt
from . import summarize as _summarize
from . import transcribe as _transcribe


def transcribe():
    """Audio file → diarized transcript as JSON lines on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-transcribe",
        description="Transcribe audio with speaker diarization.",
    )
    p.add_argument("audio", help="path to audio file")
    p.add_argument("--model", default="base",
                   help="whisper model size (default: base)")
    p.add_argument("--device", default="cpu",
                   help="compute device (default: cpu)")
    p.add_argument("--compute-type", default="int8",
                   help="quantization type (default: int8)")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="exact number of speakers")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="minimum number of speakers")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="maximum number of speakers")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    args = p.parse_args()

    segments = _transcribe.transcribe_and_diarize(
        args.audio,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
    )
    print(_transcribe.to_jsonl(segments))


def fmt():
    """JSON lines on stdin → formatted transcript on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-fmt",
        description="Format diarized transcript as readable text.",
    )
    p.add_argument("--timestamps", action="store_true",
                   help="include timestamps")
    args = p.parse_args()

    text = sys.stdin.read()
    if not text.strip():
        return
    segments = _transcribe.from_jsonl(text)
    print(_fmt.fmt(segments, timestamps=args.timestamps))


def summarize():
    """JSON lines on stdin + template → summary on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-summarize",
        description="Summarize a transcript using a template and LLM.",
    )
    p.add_argument("--template", required=True,
                   help="path to template file")
    p.add_argument("--base-url", default=None,
                   help="LLM API base URL (or set MN_API_BASE)")
    p.add_argument("--model", default=None,
                   help="LLM model name (or set MN_MODEL)")
    p.add_argument("--api-key", default=None,
                   help="API key (or set MN_API_KEY)")
    args = p.parse_args()

    text = sys.stdin.read()
    if not text.strip():
        return
    segments = _transcribe.from_jsonl(text)
    result = _summarize.summarize(
        segments,
        args.template,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )
    print(result)


def main():
    """Composed pipeline: audio file → note on stdout."""
    p = argparse.ArgumentParser(
        prog="mn",
        description="Transcribe, diarize, and summarize audio to notes.",
    )
    p.add_argument("audio", help="path to audio file")
    p.add_argument("--template", required=True,
                   help="path to template file")
    # whisper options
    p.add_argument("--model", default="base",
                   help="whisper model size (default: base)")
    p.add_argument("--device", default="cpu",
                   help="compute device (default: cpu)")
    p.add_argument("--compute-type", default="int8",
                   help="quantization type (default: int8)")
    # diarization options
    p.add_argument("--num-speakers", type=int, default=None,
                   help="exact number of speakers")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="minimum number of speakers")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="maximum number of speakers")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # llm options
    p.add_argument("--base-url", default=None,
                   help="LLM API base URL (or set MN_API_BASE)")
    p.add_argument("--llm-model", default=None,
                   help="LLM model name (or set MN_MODEL)")
    p.add_argument("--api-key", default=None,
                   help="API key (or set MN_API_KEY)")
    # output mode
    p.add_argument("--transcript-only", action="store_true",
                   help="print formatted transcript, skip summarization")
    args = p.parse_args()

    segments = _transcribe.transcribe_and_diarize(
        args.audio,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
    )

    if args.transcript_only:
        print(_fmt.fmt(segments, timestamps=True))
        return

    result = _summarize.summarize(
        segments,
        args.template,
        base_url=args.base_url,
        model=args.llm_model,
        api_key=args.api_key,
    )
    print(result)
