"""Tests for mn.log — structured logging with verbosity control."""

from mn import log as _log


class TestLogging:

    def test_error_always_shown(self, capsys):
        _log.configure(verbose=0)
        _log.error("test error")
        err = capsys.readouterr().err
        assert "test error" in err

    def test_warn_hidden_at_verbose_0(self, capsys):
        _log.configure(verbose=0)
        _log.warn("test warning")
        err = capsys.readouterr().err
        assert "test warning" not in err

    def test_warn_shown_at_verbose_1(self, capsys):
        _log.configure(verbose=1)
        _log.warn("test warning")
        err = capsys.readouterr().err
        assert "test warning" in err

    def test_progress_hidden_at_verbose_1(self, capsys):
        _log.configure(verbose=1)
        _log.progress("test progress")
        err = capsys.readouterr().err
        assert "test progress" not in err

    def test_progress_shown_at_verbose_2(self, capsys):
        _log.configure(verbose=2)
        _log.progress("test progress")
        err = capsys.readouterr().err
        assert "test progress" in err

    def test_configure_reads_env(self, monkeypatch, capsys):
        monkeypatch.setenv("MN_VERBOSE", "0")
        _log.configure()
        _log.warn("should be hidden")
        err = capsys.readouterr().err
        assert "should be hidden" not in err

    def test_configure_defaults_to_verbose_1(self, monkeypatch, capsys):
        monkeypatch.delenv("MN_VERBOSE", raising=False)
        _log.configure()
        _log.warn("should be shown")
        err = capsys.readouterr().err
        assert "should be shown" in err

    def test_configure_clamps_high_values(self, capsys):
        _log.configure(verbose=99)
        _log.progress("test progress")
        err = capsys.readouterr().err
        assert "test progress" in err
