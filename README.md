# make-notes

Speech-to-notes pipeline for psychotherapy sessions. Records, transcribes with speaker diarization, and summarizes into structured clinical notes using plain-text templates.

Four small tools that compose via stdin/stdout:

```
audio → mn-transcribe → jsonl → mn-fmt → text
                              → mn-summarize → note
```

## Install

```sh
pip install .
```

Requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). Accept the model terms, then:

```sh
export HF_TOKEN=hf_...
```

## Usage

### One command

```sh
mn session.wav --template templates/soap.txt --num-speakers 2
```

### Composable pipeline

Each tool reads stdin, writes stdout. JSON lines as the interchange format.

```sh
# Transcribe with diarization → JSON lines
mn-transcribe session.wav --num-speakers 2 > transcript.jsonl

# Format as readable text
mn-fmt < transcript.jsonl
mn-fmt --timestamps < transcript.jsonl

# Summarize with a template
mn-summarize --template templates/soap.txt < transcript.jsonl
```

Save intermediate artifacts, re-summarize with different templates, pipe into other tools:

```sh
mn-transcribe session.wav > session.jsonl
mn-summarize --template templates/soap.txt < session.jsonl > soap.txt
mn-summarize --template templates/birp.txt < session.jsonl > birp.txt
```

### Transcript only

```sh
mn session.wav --template templates/soap.txt --transcript-only
```

## Templates

Templates are plain text files in `templates/` with `$`-placeholders:

- `$transcript` — the full timestamped transcript
- `$speakers` — comma-separated speaker list

Included templates:

| File | Format |
|------|--------|
| `soap.txt` | Subjective, Objective, Assessment, Plan |
| `dap.txt` | Data, Assessment, Plan |
| `birp.txt` | Behavior, Intervention, Response, Plan |
| `progress.txt` | General progress note |

Write your own — any text file with `$transcript` works.

## LLM Configuration

Summarization uses any OpenAI-compatible chat completions API. Defaults to local [ollama](https://ollama.com).

| Env var | Default | Description |
|---------|---------|-------------|
| `MN_API_BASE` | `http://localhost:11434/v1` | API base URL |
| `MN_MODEL` | `llama3` | Model name |
| `MN_API_KEY` | `ollama` | API key |

Or pass `--base-url`, `--llm-model`, `--api-key` flags.

Examples:

```sh
# ollama (default)
ollama serve &
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

## Diarization Options

| Flag | Description |
|------|-------------|
| `--num-speakers N` | Exact number of speakers |
| `--min-speakers N` | Minimum speakers |
| `--max-speakers N` | Maximum speakers |

For therapy sessions, `--num-speakers 2` is typical.

## Privacy

This tool processes sensitive clinical data. Keep recordings and transcripts local. Do not send session audio or transcripts to cloud APIs unless your practice has appropriate BAAs in place. Local inference with ollama + faster-whisper keeps everything on your machine.

## Architecture

```
mn/
├── transcribe.py   audio → words → speaker segments → aligned Segments
├── fmt.py          Segments → readable text
├── summarize.py    Segments + template → LLM prompt → note
└── cli.py          arg parsing, composition, stdin/stdout wiring
templates/
├── soap.txt        $transcript → SOAP note
├── dap.txt         $transcript → DAP note
├── birp.txt        $transcript → BIRP note
└── progress.txt    $transcript → general progress note
```

Each module exports pure functions. The CLI wires them together.

## Dependencies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-optimized Whisper inference
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [httpx](https://github.com/encode/httpx) — HTTP client for LLM API
