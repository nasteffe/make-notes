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

Save intermediate artifacts, re-summarize with different templates:

```sh
mn-transcribe session.wav --speakers Therapist,Client > session.jsonl
mn-summarize --template templates/soap.txt < session.jsonl > soap.txt
mn-summarize --template templates/cbt-soap.txt < session.jsonl > cbt.txt
mn-summarize --template templates/birp.txt < session.jsonl > birp.txt
```

### Redaction + summarization

```sh
# Redact as a pipeline stage
mn-transcribe session.wav | mn-redact --names "John Doe" | mn-summarize --template templates/soap.txt

# Or use the built-in --redact flag
mn session.wav --template templates/soap.txt --redact --redact-names "John Doe"
```

### Batch processing

```sh
# Process all WAV files in a directory
mn-batch sessions/ --template templates/soap.txt --speakers Therapist,Client

# Output to a separate directory
mn-batch sessions/ --template templates/soap.txt --output-dir notes/

# Transcripts only (no LLM)
mn-batch sessions/ --template templates/soap.txt --transcript-only
```

### Recording

```sh
# Record until Ctrl-C
mn-record -o session.wav

# Record with a time limit
mn-record -o session.wav --duration 3600

# Record to a temp file, print the path
mn-record
```

### Browse templates

```sh
mn-templates
mn-templates --dir ./my-templates/
```

## Templates

Templates are plain text files with `$`-placeholders:

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
| `intake.txt` | Intake assessment |

Write your own — any text file with `$transcript` works.

## LLM Configuration

Summarization uses any OpenAI-compatible chat completions API. Defaults to local [ollama](https://ollama.com). See `Modelfile` for recommended models.

| Env var | Default | Description |
|---------|---------|-------------|
| `MN_API_BASE` | `http://localhost:11434/v1` | API base URL |
| `MN_MODEL` | `llama3` | Model name |
| `MN_API_KEY` | `ollama` | API key |

Or pass `--base-url`, `--llm-model`, `--api-key` flags.

```sh
# ollama (default)
ollama pull llama3.1:8b
mn session.wav --template templates/soap.txt

# openai
export MN_API_BASE=https://api.openai.com/v1
export MN_MODEL=gpt-4o
export MN_API_KEY=sk-...

# any openai-compatible endpoint
export MN_API_BASE=https://api.together.xyz/v1
export MN_MODEL=meta-llama/Llama-3-70b-chat-hf
export MN_API_KEY=...
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

- **Local by default**: ollama + faster-whisper keeps everything on your machine.
- **mn-redact**: Strip names, phone numbers, SSNs, emails, and addresses before sending transcripts to any LLM.
- **--redact flag**: Built into `mn`, `mn-summarize`, and `mn-batch` for convenience.
- Do not send session audio or transcripts to cloud APIs unless your practice has appropriate BAAs in place.

## Architecture

```
mn/
├── transcribe.py   audio → words → speaker segments → aligned Segments
├── fmt.py          Segments → readable text
├── record.py       microphone → WAV file
├── redact.py       Segments → PII-redacted Segments
├── edit.py         Segments → $EDITOR → corrected Segments
├── summarize.py    Segments + template → LLM prompt → note
└── cli.py          arg parsing, composition, stdin/stdout wiring
templates/
├── soap.txt        SOAP note
├── dap.txt         DAP note
├── birp.txt        BIRP note
├── progress.txt    General progress note
├── cbt-soap.txt    CBT-oriented SOAP note
├── psychodynamic.txt  Psychodynamic process note
└── intake.txt      Intake assessment
Modelfile           Recommended ollama models for clinical use
```

Each module exports pure functions. The CLI wires them together.

## Dependencies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-optimized Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [httpx](https://github.com/encode/httpx) — HTTP client for LLM API
- [sounddevice](https://github.com/spatialaudio/python-sounddevice) + [soundfile](https://github.com/bastibe/python-soundfile) — microphone recording (optional)
