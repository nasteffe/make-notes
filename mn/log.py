"""Structured logging with verbosity control.

All stderr output goes through this module so users can control noise:

    MN_VERBOSE=0   only errors (quiet)
    MN_VERBOSE=1   errors + warnings  (default)
    MN_VERBOSE=2   errors + warnings + progress

CLI tools call configure() once at startup. Library code uses the
module-level functions: error(), warn(), progress().
"""

import logging
import os
import sys

_logger = logging.getLogger("mn")

_PROGRESS = 15  # between DEBUG(10) and INFO(20)
logging.addLevelName(_PROGRESS, "PROGRESS")

# Map MN_VERBOSE values → logging level thresholds.
_LEVEL_MAP = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: _PROGRESS,
}


class _StderrHandler(logging.StreamHandler):
    """Handler that writes plain messages to stderr (no timestamp/level prefix).

    Always resolves sys.stderr at emit time so that test frameworks like
    pytest that monkeypatch sys.stderr can capture output.
    """

    def __init__(self):
        # Don't lock to a specific stream at init time.
        super().__init__(sys.stderr)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        # Resolve current sys.stderr each time, not the one captured at init.
        self.stream = sys.stderr
        super().emit(record)


def configure(verbose=None):
    """Set up the mn logger. Call once from CLI entry points.

    verbose: 0, 1, or 2.  None → read MN_VERBOSE env (default 1).
    """
    if verbose is None:
        try:
            verbose = int(os.environ.get("MN_VERBOSE", "1"))
        except ValueError:
            verbose = 1
    verbose = max(0, min(2, verbose))

    _logger.handlers.clear()
    _logger.addHandler(_StderrHandler())
    _logger.setLevel(_LEVEL_MAP.get(verbose, logging.WARNING))
    _logger.propagate = False


# Auto-configure with defaults so messages work even without explicit
# configure() call (e.g. in library usage or tests).
configure()


def error(msg):
    """Always shown (all verbosity levels)."""
    _logger.error(msg)


def warn(msg):
    """Shown at verbosity >= 1 (the default)."""
    _logger.warning(msg)


def progress(msg):
    """Shown at verbosity >= 2."""
    _logger.log(_PROGRESS, msg)
