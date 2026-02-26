"""CLI entry points for make-notes.

Nine tools, each useful alone, composable together:

    mn-record > session.wav
    mn-transcribe session.wav --speakers Therapist,Client > transcript.jsonl
    mn-redact < transcript.jsonl > redacted.jsonl
    mn-fmt < transcript.jsonl
    mn-edit < transcript.jsonl > corrected.jsonl
    mn-summarize --template templates/soap.txt < transcript.jsonl
    mn-templates
    mn-batch sessions/ --template templates/soap.txt
    mn session.wav --template templates/soap.txt

Stdin/stdout everywhere. JSON lines as the interchange format.
"""

import argparse
import os
import sys
from pathlib import Path

import httpx

from . import edit as _edit
from . import fmt as _fmt
from . import log as _log
from . import record as _record
from . import redact as _redact
from . import summarize as _summarize
from . import transcribe as _transcribe
from .config import apply_config, load_config
from .summarize import RemoteEndpointError


# -- Error handling ---------------------------------------------------------


def _die(msg):
    """Print an error message to stderr and exit with code 1."""
    _log.error(f"Error: {msg}")
    sys.exit(1)


def _check_audio_file(path):
    """Validate that an audio file exists and is readable."""
    p = Path(path)
    if not p.exists():
        _die(f"Audio file not found: {path}")
    if not p.is_file():
        _die(f"Not a file: {path}")
    if p.stat().st_size == 0:
        _die(f"Audio file is empty: {path}")


def _check_hf_token():
    """Verify that the HF_TOKEN environment variable is set."""
    if not os.environ.get("HF_TOKEN"):
        _die(
            "HuggingFace token required for speaker diarization.\n"
            "Set the HF_TOKEN environment variable.\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )


def _check_template(path):
    """Validate that a template file exists."""
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        _die(f"Template file not found: {path}")
    if not p.is_file():
        _die(f"Not a file: {path}")


# -- Progress reporting ----------------------------------------------------


def _progress(msg):
    """Print a progress message to stderr (verbosity >= 2)."""
    _log.progress(msg)


# -- Shared argument helpers ------------------------------------------------


def _add_verbose_flag(p):
    """Add --verbose / -v flag (repeatable) to control stderr output."""
    p.add_argument("-v", "--verbose", action="count", default=None,
                   help="increase output verbosity (repeat for more: -v warnings, -vv progress)")


def _init_logging(args):
    """Configure logging from the parsed --verbose flag."""
    level = getattr(args, "verbose", None)
    if level is None:
        _log.configure()  # use MN_VERBOSE env
    else:
        _log.configure(verbose=min(level, 2))


def _add_whisper_args(p):
    p.add_argument("--model", default="base",
                   help="whisper model size (default: base)")
    p.add_argument("--device", default="cpu",
                   help="compute device (default: cpu)")
    p.add_argument("--compute-type", default="int8",
                   help="quantization type (default: int8)")


def _add_diarization_args(p):
    p.add_argument("--num-speakers", type=int, default=None,
                   help="exact number of speakers")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="minimum number of speakers")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="maximum number of speakers")
    p.add_argument("--speakers", default=None,
                   help="comma-separated speaker names (e.g. Therapist,Client)")


def _add_llm_args(p):
    p.add_argument("--base-url", default=None,
                   help="LLM API base URL (or set MN_API_BASE)")
    p.add_argument("--llm-model", default=None,
                   help="LLM model name (or set MN_MODEL)")
    p.add_argument("--api-key", default=None,
                   help="API key (or set MN_API_KEY)")
    p.add_argument("--allow-remote", action="store_true",
                   help="allow sending transcript to a non-local LLM endpoint")
    p.add_argument("--client-name", default=None,
                   help="client name for template $client_name placeholder")
    p.add_argument("--session-date", default=None,
                   help="session date for template $date placeholder (YYYY-MM-DD)")


def _transcribe_audio(args, _whisper=None, _diarizer=None, quiet=False):
    """Shared transcription logic for transcribe and main commands.

    Pass _whisper/_diarizer to reuse pre-loaded models (batch mode).
    """
    if not quiet:
        _progress("Transcribing...")
    try:
        segments = _transcribe.transcribe_and_diarize(
            args.audio,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            _whisper=_whisper,
            _diarizer=_diarizer,
        )
    except (OSError, RuntimeError, ValueError) as e:
        _die(f"Transcription failed: {e}")

    if args.speakers:
        segments = _transcribe.label_speakers(segments, args.speakers)

    if not quiet:
        n = len(segments)
        speakers = len(set(s.speaker for s in segments))
        _progress(f"  {n} segments, {speakers} speaker(s)")
    return segments


def _read_stdin_segments():
    """Read JSONL segments from stdin, return None if empty."""
    text = sys.stdin.read()
    if not text.strip():
        return None
    return _transcribe.from_jsonl(text)


def _apply_redaction(segments, args):
    """Apply PII redaction if --redact was passed. Returns (possibly new) list."""
    if not args.redact:
        return segments
    names = ([n.strip() for n in args.redact_names.split(",")]
             if args.redact_names else None)
    return _redact.redact(segments, extra_names=names)


def _call_summarize(segments, args):
    """Call the LLM summarize pipeline, handling common errors."""
    _progress("Summarizing...")
    try:
        return _summarize.summarize(
            segments,
            args.template,
            client_name=args.client_name,
            session_date=args.session_date,
            base_url=args.base_url,
            model=args.llm_model,
            api_key=args.api_key,
            allow_remote=args.allow_remote,
        )
    except RemoteEndpointError as e:
        _die(str(e))
    except httpx.ConnectError:
        _die(
            "Could not connect to LLM endpoint. Is ollama running?\n"
            "Start it with: ollama serve\n"
            "Or set MN_API_BASE to a different endpoint."
        )
    except httpx.HTTPStatusError as e:
        _die(f"LLM request failed: {e.response.status_code} {e.response.text}")


# -- mn-record --------------------------------------------------------------


def record():
    """Record audio from microphone → WAV file path on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-record",
        description="Record audio from microphone to a WAV file.",
    )
    p.add_argument("-o", "--output", default=None,
                   help="output WAV path (default: temp file)")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="sample rate in Hz (default: 16000)")
    p.add_argument("--channels", type=int, default=1,
                   help="number of channels (default: 1)")
    p.add_argument("--duration", type=float, default=None,
                   help="max duration in seconds (default: until Ctrl-C)")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)

    path = _record.record(
        output_path=args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        duration=args.duration,
    )
    if path is None:
        _die("No audio captured.")
    print(path)


# -- mn-transcribe ----------------------------------------------------------


def transcribe():
    """Audio file → diarized transcript as JSON lines on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-transcribe",
        description="Transcribe audio with speaker diarization.",
    )
    p.add_argument("audio", help="path to audio file")
    _add_whisper_args(p)
    _add_diarization_args(p)
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)
    apply_config(args, load_config())

    _check_audio_file(args.audio)
    _check_hf_token()

    segments = _transcribe_audio(args)
    print(_transcribe.to_jsonl(segments))


# -- mn-fmt -----------------------------------------------------------------


def fmt():
    """JSON lines on stdin → formatted transcript on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-fmt",
        description="Format diarized transcript as readable text.",
    )
    p.add_argument("--timestamps", action="store_true",
                   help="include timestamps")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)

    segments = _read_stdin_segments()
    if segments is None:
        return
    print(_fmt.fmt(segments, timestamps=args.timestamps))


# -- mn-redact --------------------------------------------------------------


def redact():
    """JSON lines on stdin → redacted JSON lines on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-redact",
        description="Redact PII from transcript segments.",
    )
    p.add_argument("--names", default=None,
                   help="comma-separated names to redact")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)

    segments = _read_stdin_segments()
    if segments is None:
        return

    names = [n.strip() for n in args.names.split(",")] if args.names else None
    redacted = _redact.redact(segments, extra_names=names)
    print(_transcribe.to_jsonl(redacted))


# -- mn-edit ----------------------------------------------------------------


def edit():
    """JSON lines on stdin → open in $EDITOR → corrected JSON lines on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-edit",
        description="Edit transcript in $EDITOR, output corrected JSON lines.",
    )
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)

    segments = _read_stdin_segments()
    if segments is None:
        return

    corrected = _edit.edit(segments)
    print(_transcribe.to_jsonl(corrected))


# -- mn-summarize -----------------------------------------------------------


def summarize():
    """JSON lines on stdin + template → summary on stdout."""
    p = argparse.ArgumentParser(
        prog="mn-summarize",
        description="Summarize a transcript using a template and LLM.",
    )
    p.add_argument("--template", required=True,
                   help="path to template file")
    _add_llm_args(p)
    p.add_argument("--redact", action="store_true",
                   help="redact PII before sending to LLM")
    p.add_argument("--redact-names", default=None,
                   help="comma-separated names to redact")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)
    apply_config(args, load_config())

    _check_template(args.template)

    segments = _read_stdin_segments()
    if segments is None:
        return

    segments = _apply_redaction(segments, args)
    print(_call_summarize(segments, args))


# -- mn-templates -----------------------------------------------------------


def templates():
    """List available templates with descriptions."""
    p = argparse.ArgumentParser(
        prog="mn-templates",
        description="List available note templates.",
    )
    p.add_argument("--dir", default=None,
                   help="template directory (default: built-in templates/)")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)

    if args.dir:
        template_dir = Path(args.dir)
    else:
        template_dir = Path(__file__).parent.parent / "templates"

    if not template_dir.is_dir():
        print(f"Template directory not found: {template_dir}", file=sys.stderr)
        sys.exit(1)

    for path in sorted(template_dir.glob("*.txt")):
        # Extract first non-empty line as description.
        desc = ""
        for line in path.read_text().split("\n"):
            line = line.strip()
            if line:
                desc = line[:72]
                break
        print(f"  {path.stem:20s} {desc}")


# -- mn-batch ---------------------------------------------------------------


def batch():
    """Process a directory of audio files → one note per file."""
    p = argparse.ArgumentParser(
        prog="mn-batch",
        description="Batch-process audio files into notes.",
    )
    p.add_argument("directory", help="directory containing audio files")
    p.add_argument("--template", required=True,
                   help="path to template file")
    p.add_argument("--ext", default=".wav",
                   help="audio file extension to match (default: .wav)")
    p.add_argument("--output-dir", default=None,
                   help="output directory for notes (default: same as input)")
    _add_whisper_args(p)
    _add_diarization_args(p)
    _add_llm_args(p)
    p.add_argument("--redact", action="store_true",
                   help="redact PII before sending to LLM")
    p.add_argument("--redact-names", default=None,
                   help="comma-separated names to redact")
    p.add_argument("--transcript-only", action="store_true",
                   help="output transcripts only, skip summarization")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)
    apply_config(args, load_config())

    _check_template(args.template)
    _check_hf_token()

    audio_dir = Path(args.directory)
    files = sorted(audio_dir.glob(f"*{args.ext}"))

    if not files:
        _die(f"No {args.ext} files in {audio_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else audio_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load models once for the whole batch.
    _progress("Loading models...")
    try:
        whisper = _transcribe.load_whisper(
            args.model, args.device, args.compute_type,
        )
        diarizer = _transcribe.load_diarizer()
    except (OSError, RuntimeError, ValueError) as e:
        _die(f"Failed to load models: {e}")

    failed = []
    for i, audio_path in enumerate(files, 1):
        _progress(f"[{i}/{len(files)}] {audio_path.name}")
        _check_audio_file(audio_path)
        args.audio = str(audio_path)

        try:
            segments = _transcribe_audio(
                args, _whisper=whisper, _diarizer=diarizer, quiet=True,
            )

            segments = _apply_redaction(segments, args)

            if args.transcript_only:
                out_file = output_dir / f"{audio_path.stem}.txt"
                out_file.write_text(_fmt.fmt(segments, timestamps=True) + "\n")
            else:
                result = _call_summarize(segments, args)
                out_file = output_dir / f"{audio_path.stem}.note.txt"
                out_file.write_text(result + "\n")
        except SystemExit:
            failed.append(audio_path.name)
            continue

        _progress(f"  → {out_file}")

    if failed:
        _progress(f"Warning: {len(failed)} file(s) failed: {', '.join(failed)}")
    _progress(f"Done. {len(files) - len(failed)}/{len(files)} file(s) processed.")


# -- mn (main) --------------------------------------------------------------


def main():
    """Composed pipeline: audio file → note on stdout."""
    p = argparse.ArgumentParser(
        prog="mn",
        description="Transcribe, diarize, and summarize audio to notes.",
    )
    p.add_argument("audio", help="path to audio file")
    p.add_argument("--template", required=True,
                   help="path to template file")
    _add_whisper_args(p)
    _add_diarization_args(p)
    _add_llm_args(p)
    p.add_argument("--redact", action="store_true",
                   help="redact PII before sending to LLM")
    p.add_argument("--redact-names", default=None,
                   help="comma-separated names to redact")
    p.add_argument("--transcript-only", action="store_true",
                   help="print formatted transcript, skip summarization")
    _add_verbose_flag(p)
    args = p.parse_args()
    _init_logging(args)
    apply_config(args, load_config())

    _check_audio_file(args.audio)
    _check_template(args.template)
    _check_hf_token()

    segments = _transcribe_audio(args)
    segments = _apply_redaction(segments, args)

    if args.transcript_only:
        print(_fmt.fmt(segments, timestamps=True))
        return

    print(_call_summarize(segments, args))
