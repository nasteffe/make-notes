"""Tests for mn.edit — round-trip editable transcript format."""

import os
import stat
import subprocess
from unittest.mock import patch

import pytest

from mn.edit import _parse_time, edit, from_editable, to_editable
from mn.transcribe import Segment


def _sample():
    return [
        Segment("SPEAKER_00", "I've been feeling anxious.", 0.0, 65.0),
        Segment("SPEAKER_01", "Can you tell me more?", 70.0, 125.0),
    ]


# -- to_editable() ----------------------------------------------------------


class TestToEditable:

    def test_basic_format(self):
        result = to_editable(_sample())
        assert "[00:00 → 01:05] SPEAKER_00:" in result
        assert "I've been feeling anxious." in result
        assert "[01:10 → 02:05] SPEAKER_01:" in result
        assert "Can you tell me more?" in result

    def test_blocks_separated_by_blank_lines(self):
        result = to_editable(_sample())
        blocks = result.strip().split("\n\n")
        assert len(blocks) == 2

    def test_ends_with_newline(self):
        result = to_editable(_sample())
        assert result.endswith("\n")

    def test_single_segment(self):
        result = to_editable([Segment("A", "hello", 0.0, 5.0)])
        assert "[00:00 → 00:05] A:" in result
        assert "hello" in result

    def test_empty(self):
        assert to_editable([]) == "\n"


# -- from_editable() --------------------------------------------------------


class TestFromEditable:

    def test_parses_basic(self):
        text = (
            "[00:00 → 01:05] SPEAKER_00:\n"
            "I've been feeling anxious.\n"
            "\n"
            "[01:10 → 02:05] SPEAKER_01:\n"
            "Can you tell me more?\n"
        )
        result = from_editable(text)
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[0].text == "I've been feeling anxious."
        assert result[0].start == 0.0
        assert result[0].end == 65.0
        assert result[1].speaker == "SPEAKER_01"
        assert result[1].start == 70.0
        assert result[1].end == 125.0

    def test_preserves_edited_speaker_names(self):
        text = (
            "[00:00 → 00:05] Therapist:\n"
            "How are you?\n"
        )
        result = from_editable(text)
        assert result[0].speaker == "Therapist"

    def test_preserves_edited_text(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "Corrected transcription text.\n"
        )
        result = from_editable(text)
        assert result[0].text == "Corrected transcription text."

    def test_multiline_body(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "First line.\n"
            "Second line.\n"
        )
        result = from_editable(text)
        assert "First line.\nSecond line." == result[0].text

    def test_body_with_blank_lines(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "First paragraph.\n"
            "\n"
            "Second paragraph.\n"
            "\n"
            "[00:05 → 00:10] B:\n"
            "Next speaker.\n"
        )
        result = from_editable(text)
        assert len(result) == 2
        assert "First paragraph.\n\nSecond paragraph." == result[0].text
        assert result[1].text == "Next speaker."

    def test_body_with_multiple_blank_lines(self):
        text = (
            "[00:00 → 00:05] A:\n"
            "Line one.\n"
            "\n"
            "\n"
            "\n"
            "Line two.\n"
        )
        result = from_editable(text)
        assert len(result) == 1
        assert "Line one.\n\n\n\nLine two." == result[0].text

    def test_empty_string(self):
        assert from_editable("") == []

    def test_skips_malformed_blocks(self):
        text = (
            "This is not a valid header\n"
            "some text\n"
            "\n"
            "[00:00 → 00:05] A:\n"
            "Valid block.\n"
        )
        result = from_editable(text)
        assert len(result) == 1
        assert result[0].speaker == "A"


# -- Round-trip -------------------------------------------------------------


class TestRoundTrip:

    def test_roundtrip_preserves_content(self):
        original = _sample()
        text = to_editable(original)
        restored = from_editable(text)
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig.speaker == rest.speaker
            assert orig.text == rest.text
            # Timestamps lose sub-second precision through MM:SS format
            assert abs(orig.start - rest.start) < 60
            assert abs(orig.end - rest.end) < 60

    def test_roundtrip_single_segment(self):
        original = [Segment("Solo", "just me talking", 30.0, 90.0)]
        restored = from_editable(to_editable(original))
        assert restored[0].speaker == "Solo"
        assert restored[0].text == "just me talking"


# -- _parse_time() ----------------------------------------------------------


class TestParseTime:

    def test_zero(self):
        assert _parse_time("0:00") == 0

    def test_seconds(self):
        assert _parse_time("0:45") == 45

    def test_minutes_and_seconds(self):
        assert _parse_time("2:05") == 125

    def test_large_minutes(self):
        assert _parse_time("61:01") == 3661

    def test_invalid_single_part(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            _parse_time("123")

    def test_invalid_three_parts(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            _parse_time("1:2:3")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            _parse_time("a:b")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            _parse_time("")


# -- edit() (subprocess / editor launcher) ----------------------------------


class TestEditFunction:

    def _segments(self):
        return [Segment("A", "Original text.", 0.0, 5.0)]

    def test_happy_path_returns_edited_segments(self, monkeypatch, tmp_path):
        """Editor modifies the text; edit() returns the corrected segments."""
        # Use a real script as the editor: replaces "Original" with "Edited".
        script = tmp_path / "fake-editor.sh"
        script.write_text('#!/bin/sh\nsed -i "s/Original/Edited/" "$1"\n')
        script.chmod(0o755)

        monkeypatch.setenv("EDITOR", str(script))
        result = edit(self._segments())
        assert len(result) == 1
        assert result[0].text == "Edited text."

    def test_editor_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nonexistent-editor-xyz")
        with pytest.raises(FileNotFoundError, match="not found"):
            edit(self._segments())

    def test_editor_failure_raises(self, monkeypatch, tmp_path):
        """Non-zero exit from the editor should propagate as CalledProcessError."""
        script = tmp_path / "fail-editor.sh"
        script.write_text("#!/bin/sh\nexit 1\n")
        script.chmod(0o755)

        monkeypatch.setenv("EDITOR", str(script))
        with pytest.raises(subprocess.CalledProcessError):
            edit(self._segments())

    def test_temp_file_is_private(self, monkeypatch, tmp_path):
        """Temp file should be created with mode 0o600 (owner-only)."""
        script = tmp_path / "check-perms.sh"
        # The editor just reads the file — we inspect perms before it runs.
        script.write_text("#!/bin/sh\ntrue\n")
        script.chmod(0o755)

        permissions = []
        original_run = subprocess.run

        def spy_run(cmd, **kwargs):
            # cmd[1] is the temp file path
            mode = os.stat(cmd[1]).st_mode
            permissions.append(stat.S_IMODE(mode))
            return original_run(cmd, **kwargs)

        monkeypatch.setenv("EDITOR", str(script))
        with patch("mn.edit.subprocess.run", side_effect=spy_run):
            edit(self._segments())

        assert permissions[0] == 0o600

    def test_temp_files_cleaned_up(self, monkeypatch, tmp_path):
        """Temp dir and file should be removed after edit() returns."""
        script = tmp_path / "noop-editor.sh"
        script.write_text("#!/bin/sh\ntrue\n")
        script.chmod(0o755)

        created_paths = []
        original_run = subprocess.run

        def spy_run(cmd, **kwargs):
            created_paths.append(cmd[1])
            return original_run(cmd, **kwargs)

        monkeypatch.setenv("EDITOR", str(script))
        with patch("mn.edit.subprocess.run", side_effect=spy_run):
            edit(self._segments())

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])
        assert not os.path.exists(os.path.dirname(created_paths[0]))

    def test_cleanup_on_editor_failure(self, monkeypatch, tmp_path):
        """Temp files should be cleaned up even when the editor fails."""
        script = tmp_path / "fail-editor.sh"
        script.write_text("#!/bin/sh\nexit 1\n")
        script.chmod(0o755)

        created_paths = []
        original_run = subprocess.run

        def spy_run(cmd, **kwargs):
            created_paths.append(cmd[1])
            return original_run(cmd, **kwargs)

        monkeypatch.setenv("EDITOR", str(script))
        with pytest.raises(subprocess.CalledProcessError):
            with patch("mn.edit.subprocess.run", side_effect=spy_run):
                edit(self._segments())

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])
        assert not os.path.exists(os.path.dirname(created_paths[0]))
