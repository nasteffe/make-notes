"""Tests for mn.config — TOML config file loading and application."""

import argparse
from pathlib import Path

import pytest

from mn.config import apply_config, find_config, load_config


# -- find_config() ----------------------------------------------------------


class TestFindConfig:

    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert find_config() is None

    def test_finds_project_local_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "mn.toml"
        cfg.write_text("[transcribe]\nmodel = 'large-v3'\n")
        result = find_config()
        assert result is not None
        assert result.resolve() == cfg.resolve()

    def test_finds_xdg_config(self, tmp_path, monkeypatch):
        subdir = tmp_path / "elsewhere"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        xdg = tmp_path / "xdg"
        xdg.mkdir()
        cfg = xdg / "mn.toml"
        cfg.write_text("[transcribe]\nmodel = 'tiny'\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        result = find_config()
        assert result is not None
        assert result.resolve() == cfg.resolve()

    def test_project_local_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "mn.toml"
        local.write_text("[transcribe]\nmodel = 'local'\n")
        xdg = tmp_path / "xdg"
        xdg.mkdir()
        (xdg / "mn.toml").write_text("[transcribe]\nmodel = 'xdg'\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        result = find_config()
        assert result is not None
        assert result.resolve() == local.resolve()


# -- load_config() ----------------------------------------------------------


class TestLoadConfig:

    def test_loads_toml(self, tmp_path):
        cfg = tmp_path / "mn.toml"
        cfg.write_text(
            "[transcribe]\nmodel = 'large-v3'\nnum_speakers = 2\n"
            "[summarize]\ntemplate = 'templates/soap.txt'\n"
        )
        result = load_config(cfg)
        assert result["transcribe"]["model"] == "large-v3"
        assert result["transcribe"]["num_speakers"] == 2
        assert result["summarize"]["template"] == "templates/soap.txt"

    def test_returns_empty_when_no_file(self):
        assert load_config(None) == {}

    def test_returns_empty_for_missing_path(self, tmp_path):
        assert load_config(tmp_path / "nonexistent.toml") == {}

    def test_malformed_toml_returns_empty_with_warning(self, tmp_path, capsys):
        cfg = tmp_path / "bad.toml"
        cfg.write_text("this is not valid [[ toml syntax")
        result = load_config(cfg)
        assert result == {}
        err = capsys.readouterr().err
        assert "could not parse" in err.lower()


# -- apply_config() --------------------------------------------------------


class TestApplyConfig:

    def _make_args(self, **kwargs):
        """Create an argparse Namespace with typical defaults."""
        defaults = dict(
            model=None, device=None, compute_type=None,
            num_speakers=None, min_speakers=None, max_speakers=None,
            speakers=None, hf_token=None,
            template=None, base_url=None, llm_model=None, api_key=None,
            client_name=None, session_date=None,
            redact=False, redact_names=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_fills_none_values(self):
        args = self._make_args()
        config = {"transcribe": {"model": "large-v3", "num_speakers": 2}}
        apply_config(args, config)
        assert args.model == "large-v3"
        assert args.num_speakers == 2

    def test_cli_flags_override_config(self):
        args = self._make_args(model="base")
        config = {"transcribe": {"model": "large-v3"}}
        apply_config(args, config)
        assert args.model == "base"  # CLI wins

    def test_fills_summarize_section(self):
        args = self._make_args()
        config = {"summarize": {
            "template": "templates/soap.txt",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434/v1",
        }}
        apply_config(args, config)
        assert args.template == "templates/soap.txt"
        assert args.llm_model == "llama3.1:8b"
        assert args.base_url == "http://localhost:11434/v1"

    def test_fills_redact_section(self):
        args = self._make_args()
        config = {"redact": {"enabled": True, "names": "John,Jane"}}
        apply_config(args, config)
        assert args.redact is True
        assert args.redact_names == "John,Jane"

    def test_redact_cli_flag_overrides(self):
        args = self._make_args(redact=True, redact_names="Bob")
        config = {"redact": {"enabled": False, "names": "John"}}
        apply_config(args, config)
        assert args.redact is True  # CLI already True, stays True
        assert args.redact_names == "Bob"  # CLI wins

    def test_empty_config_no_change(self):
        args = self._make_args(model="base")
        apply_config(args, {})
        assert args.model == "base"

    def test_missing_attributes_skipped(self):
        # An args namespace that only has 'model' — config keys for
        # missing attributes should be silently skipped.
        args = argparse.Namespace(model=None)
        config = {"transcribe": {"model": "large-v3", "device": "cuda"}}
        apply_config(args, config)
        assert args.model == "large-v3"
        assert not hasattr(args, "device")

    def test_speakers_from_config(self):
        args = self._make_args()
        config = {"transcribe": {"speakers": "Therapist,Client"}}
        apply_config(args, config)
        assert args.speakers == "Therapist,Client"

    def test_wrong_type_ignored_with_warning(self, capsys):
        from mn import log as _log
        _log.configure(verbose=1)

        args = self._make_args()
        config = {"transcribe": {"num_speakers": "two"}}
        apply_config(args, config)
        assert args.num_speakers is None  # not set
        err = capsys.readouterr().err
        assert "should be int" in err

    def test_correct_type_accepted(self):
        args = self._make_args()
        config = {"transcribe": {"num_speakers": 3}}
        apply_config(args, config)
        assert args.num_speakers == 3

    # -- allow_remote (boolean / store_true handling) -----------------------

    def test_allow_remote_from_config(self):
        args = self._make_args(allow_remote=False)
        config = {"summarize": {"allow_remote": True}}
        apply_config(args, config)
        assert args.allow_remote is True

    def test_allow_remote_cli_flag_overrides_config(self):
        args = self._make_args(allow_remote=True)
        config = {"summarize": {"allow_remote": False}}
        apply_config(args, config)
        assert args.allow_remote is True  # CLI wins

    def test_allow_remote_false_in_config_does_not_enable(self):
        args = self._make_args(allow_remote=False)
        config = {"summarize": {"allow_remote": False}}
        apply_config(args, config)
        assert args.allow_remote is False

    def test_allow_remote_missing_attribute_skipped(self):
        # Namespace without allow_remote (e.g. mn-transcribe)
        args = argparse.Namespace(model=None)
        config = {"summarize": {"allow_remote": True}}
        apply_config(args, config)
        assert not hasattr(args, "allow_remote")
