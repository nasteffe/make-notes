"""Load defaults from a config file so flags don't need repeating.

Searches for config in this order (first found wins):

    1. ./mn.toml          (project-local)
    2. ~/.config/mn.toml  (user-level)

Config is TOML format:

    [transcribe]
    model = "large-v3"
    device = "cuda"
    compute_type = "float16"
    num_speakers = 2
    speakers = "Therapist,Client"

    [summarize]
    template = "templates/soap.txt"
    base_url = "http://localhost:11434/v1"
    model = "llama3.1:8b"
    client_name = "Client"

    [redact]
    enabled = true
    names = "John Doe,Jane Doe"

CLI flags always override config file values.
"""

import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def _search_paths():
    """Return config search paths, evaluated at call time."""
    return [
        Path("mn.toml"),
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "mn.toml",
    ]


def find_config():
    """Return the path to the first config file found, or None."""
    for p in _search_paths():
        if p.is_file():
            return p
    return None


def load_config(path=None):
    """Load config from a TOML file. Returns a dict (empty if no file/parser)."""
    if path is None:
        path = find_config()
    if path is None:
        return {}
    if tomllib is None:
        return {}

    path = Path(path)
    if not path.is_file():
        return {}

    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_config(args, config):
    """Apply config defaults to an argparse Namespace.

    Only sets values that are still at their argparse defaults (None or False).
    CLI flags always win.
    """
    transcribe_cfg = config.get("transcribe", {})
    summarize_cfg = config.get("summarize", {})
    redact_cfg = config.get("redact", {})

    # Map config keys → argparse attribute names.
    _apply_section(args, transcribe_cfg, {
        "model": "model",
        "device": "device",
        "compute_type": "compute_type",
        "num_speakers": "num_speakers",
        "min_speakers": "min_speakers",
        "max_speakers": "max_speakers",
        "speakers": "speakers",
        "hf_token": "hf_token",
    })

    _apply_section(args, summarize_cfg, {
        "template": "template",
        "base_url": "base_url",
        "model": "llm_model",
        "api_key": "api_key",
        "client_name": "client_name",
        "session_date": "session_date",
    })

    # Redaction config.
    if redact_cfg.get("enabled") and hasattr(args, "redact") and not args.redact:
        args.redact = True
    if redact_cfg.get("names") and hasattr(args, "redact_names"):
        if args.redact_names is None:
            args.redact_names = redact_cfg["names"]


def _apply_section(args, section, mapping):
    """Apply a config section to args. Only fills in None/unset values."""
    for config_key, attr_name in mapping.items():
        if not hasattr(args, attr_name):
            continue
        current = getattr(args, attr_name)
        if current is not None:
            continue
        if config_key in section:
            setattr(args, attr_name, section[config_key])
