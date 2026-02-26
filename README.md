# make-notes

Speech-to-notes pipeline for psychotherapy sessions. Records, transcribes with speaker diarization, and summarizes into structured clinical notes using plain-text templates.

Nine small tools that compose via stdin/stdout:

```
mn-record → wav → mn-transcribe → jsonl → mn-redact → jsonl
                                        → mn-edit   → jsonl
                                        → mn-fmt    → text
                                        → mn-summarize → note
mn-templates    list available templates
mn-batch        process a directory of audio files
mn              full pipeline in one command
```

## Install

```sh
pip install .

# optional: microphone recording
pip install ".[record]"
```

Requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). Accept the model terms, then:

```sh
export HF_TOKEN=hf_...
```

## Quick start

```sh
# Record, transcribe, summarize — one command
mn session.wav --template templates/soap.txt --speakers Therapist,Client --num-speakers 2

# Or record first
mn-record -o session.wav
mn session.wav --template templates/soap.txt --speakers Therapist,Client
```

## Workflow

### How the pieces fit together

The pipeline has three stages — capture, process, and output — each handled by composable tools that pass data via stdin/stdout using JSON lines (one segment per line):

```
┌─────────────┐
│  mn-record   │  mic → WAV file (optional)
└──────┬──────┘
       ▼
┌──────────────┐
│ mn-transcribe │  WAV → diarized segments (JSONL)
└──────┬───────┘
       ▼
┌──────────────────────────────────────────────┐
│  Processing (any combination, any order):      │
│    mn-edit     → correct errors in $EDITOR     │
│    mn-redact   → strip PII                     │
│    mn-fmt      → human-readable text            │
└──────┬───────────────────────────────────────┘
       ▼
┌──────────────┐
│ mn-summarize  │  JSONL + template → clinical note (via LLM)
└──────────────┘
```

Transcription is the expensive step. The JSONL interchange format lets you transcribe once and then re-process many ways without re-running whisper or diarization.

### Use case: daily clinical workflow

A therapist records sessions throughout the day, then generates notes at the end:

```sh
# During sessions — record each one
mn-record -o sessions/client_a.wav
mn-record -o sessions/client_b.wav
mn-record -o sessions/client_c.wav

# End of day — batch process everything
mn-batch sessions/ --template templates/soap.txt --speakers Therapist,Client --output-dir notes/
```

Output: `notes/client_a.note.txt`, `notes/client_b.note.txt`, etc.

### Use case: transcribe once, summarize many ways

Generate multiple note formats from a single transcription:

```sh
mn-transcribe session.wav --speakers Therapist,Client > session.jsonl

# Different note formats from the same transcript
mn-summarize --template templates/soap.txt < session.jsonl > soap.txt
mn-summarize --template templates/cbt-soap.txt < session.jsonl > cbt.txt
mn-summarize --template templates/birp.txt < session.jsonl > birp.txt
mn-summarize --template templates/progress.txt < session.jsonl > progress.txt
```

### Use case: review and correct before summarizing

Edit the transcript to fix misheard words or speaker attribution errors:

```sh
mn-transcribe session.wav --speakers Therapist,Client > raw.jsonl
mn-edit < raw.jsonl > corrected.jsonl
mn-summarize --template templates/soap.txt < corrected.jsonl
```

`mn-edit` opens the transcript in `$EDITOR` in a round-trippable format:

```
[00:00 → 00:05] Therapist:
How have you been feeling this week?

[00:05 → 00:12] Client:
I've been feeling anxious about the move.
```

Fix text, relabel speakers, then save and quit.

### Use case: privacy-first pipeline with redaction

Strip PII before any data leaves the machine:

```sh
# Pipe-based redaction
mn-transcribe session.wav | mn-redact --names "John Doe" | mn-summarize --template templates/soap.txt

# Or use built-in flags
mn session.wav --template templates/soap.txt --redact --redact-names "John Doe"
```

Redaction replaces: phone numbers → `[PHONE]`, SSNs → `[SSN]`, emails → `[EMAIL]`, dates → `[DATE]`, street addresses → `[ADDRESS]`, names → `[NAME]`.

### Use case: transcript only (no LLM)

Get a formatted transcript without running summarization:

```sh
# Single file
mn session.wav --template templates/soap.txt --transcript-only

# Batch — writes .txt files instead of .note.txt
mn-batch sessions/ --template templates/soap.txt --transcript-only --output-dir transcripts/
```

## Usage

### Composable pipeline

Each tool reads stdin, writes stdout. JSON lines as the interchange format.

```sh
# Transcribe with speaker names
mn-transcribe session.wav --num-speakers 2 --speakers Therapist,Client > transcript.jsonl

# Format as readable text
mn-fmt < transcript.jsonl
mn-fmt --timestamps < transcript.jsonl

# Edit transcript in your editor, correct errors
mn-edit < transcript.jsonl > corrected.jsonl

# Redact PII before sending to an LLM
mn-redact --names "John Doe,Jane Smith" < transcript.jsonl > redacted.jsonl

# Summarize with a template
mn-summarize --template templates/soap.txt < transcript.jsonl

# Summarize with client metadata
mn-summarize --template templates/soap.txt --client-name "J.D." --session-date 2026-02-16 < transcript.jsonl
```

### Batch processing

```sh
# Process all WAV files in a directory
mn-batch sessions/ --template templates/soap.txt --speakers Therapist,Client

# Output to a separate directory
mn-batch sessions/ --template templates/soap.txt --output-dir notes/

# Different audio format
mn-batch sessions/ --template templates/soap.txt --ext .mp3

# Transcripts only (no LLM)
mn-batch sessions/ --template templates/soap.txt --transcript-only
```

Batch mode pre-loads whisper and pyannote models once, then processes each file. If a file fails (transcription or summarization), processing continues and a summary is printed at the end.

### Recording

```sh
# Record until Ctrl-C
mn-record -o session.wav

# Record with a time limit (seconds)
mn-record -o session.wav --duration 3600

# Record to a temp file, print the path
mn-record

# Custom sample rate and channels
mn-record -o session.wav --sample-rate 44100 --channels 2
```

### Browse templates

```sh
mn-templates
mn-templates --dir ./my-templates/
```

## Configuration

Settings come from three layers. Each layer overrides the one below it:

```
CLI flags          (highest priority)
  ↑
Config file        mn.toml
  ↑
Environment vars   (lowest priority, except HF_TOKEN)
```

### Config file

Place an `mn.toml` at either location (first found wins):

1. `./mn.toml` — project-local
2. `~/.config/mn.toml` — user-level

See [`mn.toml.sample`](mn.toml.sample) for all available options with explanations. Copy it to get started:

```sh
cp mn.toml.sample mn.toml     # project-local
# or
cp mn.toml.sample ~/.config/mn.toml  # user-level
```

Example config for a therapist using a GPU with a local llama model:

```toml
[transcribe]
model = "large-v3"
device = "cuda"
compute_type = "float16"
num_speakers = 2
speakers = "Therapist,Client"

[summarize]
template = "templates/soap.txt"
model = "llama3.1:8b"

[redact]
enabled = true
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace token for pyannote diarization |
| `MN_API_BASE` | `http://localhost:11434/v1` | LLM API base URL |
| `MN_MODEL` | `llama3` | LLM model name |
| `MN_API_KEY` | `ollama` | API key for LLM endpoint |
| `MN_VERBOSE` | `1` | Logging verbosity: 0 = errors, 1 = warnings, 2 = progress |
| `EDITOR` | `vi` | Editor for `mn-edit` |

### Verbosity

Control how much output goes to stderr:

```sh
mn session.wav --template templates/soap.txt                 # default (warnings only)
mn session.wav --template templates/soap.txt -v              # + warnings
mn session.wav --template templates/soap.txt -vv             # + progress messages
MN_VERBOSE=0 mn session.wav --template templates/soap.txt    # errors only (quiet)
```

The `-v` flag works on all commands. Data always goes to stdout; logging always goes to stderr.

## Templates

Templates are plain text files with `$`-placeholders ([`string.Template`](https://docs.python.org/3/library/string.html#template-strings) syntax):

| Placeholder | Value |
|-------------|-------|
| `$transcript` | Full timestamped transcript |
| `$speakers` | Comma-separated speaker list |
| `$date` | Session date (YYYY-MM-DD, default: today) |
| `$duration` | Session duration from timestamps |
| `$client_name` | Client name (default: "Client") |

Included templates:

| File | Format |
|------|--------|
| `soap.txt` | Subjective, Objective, Assessment, Plan |
| `dap.txt` | Data, Assessment, Plan |
| `birp.txt` | Behavior, Intervention, Response, Plan |
| `progress.txt` | General progress note |
| `cbt-soap.txt` | CBT-oriented SOAP note |
| `psychodynamic.txt` | Psychodynamic process note |
| `neuropsychoanalytic.txt` | Neuropsychoanalytic process note |
| `intake.txt` | Intake assessment |
| `informed-consent.txt` | Informed consent documentation |

Write your own — any text file with `$transcript` works.

## LLM Configuration

Summarization uses any OpenAI-compatible chat completions API. Defaults to local [ollama](https://ollama.com). See `Modelfile` for recommended models.

```sh
# ollama (default — nothing to configure)
ollama pull llama3.1:8b
mn session.wav --template templates/soap.txt

# openai (requires --allow-remote or allow_remote = true in config)
export MN_API_BASE=https://api.openai.com/v1
export MN_MODEL=gpt-4o
export MN_API_KEY=sk-...
mn session.wav --template templates/soap.txt --allow-remote

# any openai-compatible endpoint
export MN_API_BASE=https://api.together.xyz/v1
export MN_MODEL=meta-llama/Llama-3-70b-chat-hf
export MN_API_KEY=...
mn session.wav --template templates/soap.txt --allow-remote
```

## Whisper Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--device` | `cpu` | Compute device (`cpu`, `cuda`) |
| `--compute-type` | `int8` | Quantization (`int8`, `float16`, `float32`) |

For clinical vocabulary, `large-v3` is significantly more accurate than `base`.

## Diarization Options

| Flag | Description |
|------|-------------|
| `--num-speakers N` | Exact number of speakers |
| `--min-speakers N` | Minimum speakers |
| `--max-speakers N` | Maximum speakers |
| `--speakers names` | Comma-separated speaker names (e.g. `Therapist,Client`) |

For therapy sessions, `--num-speakers 2 --speakers Therapist,Client` is typical.

## Privacy

This tool processes sensitive clinical data.

- **Local by default**: ollama + faster-whisper run entirely on your machine. No data leaves unless you explicitly configure a remote endpoint.
- **Remote endpoint protection**: `mn` refuses to send data to non-local LLM endpoints unless `--allow-remote` is passed or `allow_remote = true` is set in config. This prevents accidental data leakage.
- **mn-redact**: Strip names, phone numbers, SSNs, emails, and addresses before sending transcripts to any LLM.
- **--redact flag**: Built into `mn`, `mn-summarize`, and `mn-batch` for convenience.
- **Secure temp files**: `mn-edit` creates temp files with `0600` permissions in a private directory.
- Do not send session audio or transcripts to cloud APIs unless your practice has appropriate BAAs in place.

## Architecture

```
mn/
├── cli.py          arg parsing, composition, stdin/stdout wiring
├── config.py       mn.toml loading and config merging
├── log.py          structured logging with verbosity control
├── transcribe.py   audio → words → speaker segments → aligned Segments
├── fmt.py          Segments → readable text
├── record.py       microphone → WAV file
├── redact.py       Segments → PII-redacted Segments
├── edit.py         Segments → $EDITOR → corrected Segments
└── summarize.py    Segments + template → LLM prompt → note
templates/
├── soap.txt        SOAP note
├── dap.txt         DAP note
├── birp.txt        BIRP note
├── progress.txt    General progress note
├── cbt-soap.txt    CBT-oriented SOAP note
├── psychodynamic.txt  Psychodynamic process note
├── neuropsychoanalytic.txt  Neuropsychoanalytic process note
├── intake.txt      Intake assessment
└── informed-consent.txt  Informed consent documentation
mn.toml.sample      Annotated example config file
Modelfile           Recommended ollama models for clinical use
```

Each module exports pure functions. The CLI wires them together. JSONL is the interchange format between tools.

## Dependencies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-optimized Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [httpx](https://github.com/encode/httpx) — HTTP client for LLM API
- [sounddevice](https://github.com/spatialaudio/python-sounddevice) + [soundfile](https://github.com/bastibe/python-soundfile) — microphone recording (optional)
