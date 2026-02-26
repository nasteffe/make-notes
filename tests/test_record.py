"""Tests for mn.record — audio recording with mocked sounddevice/soundfile."""

import os
import signal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mn.record import record


class _FakeInputStream:
    """Mock sd.InputStream that feeds synthetic frames via the callback."""

    def __init__(self, *, samplerate, channels, callback):
        self._callback = callback
        self._channels = channels
        self._samplerate = int(samplerate)

    def __enter__(self):
        # Deliver a few frames of silence through the callback.
        chunk = np.zeros((self._samplerate, self._channels), dtype=np.float32)
        self._callback(chunk, self._samplerate, {}, None)
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture()
def _mock_sd():
    """Patch sounddevice so record() never touches real hardware."""
    mock_sd = MagicMock()
    mock_sd.InputStream = _FakeInputStream

    call_count = 0

    def fake_sleep(ms):
        nonlocal call_count
        call_count += 1
        # After the first sleep call in duration mode, just return.
        # For non-duration mode, trigger SIGINT to break the loop.
        if call_count > 1:
            os.kill(os.getpid(), signal.SIGINT)

    mock_sd.sleep = fake_sleep
    with patch.dict("sys.modules", {"sounddevice": mock_sd}):
        yield mock_sd


class TestRecord:

    def test_returns_wav_path(self, _mock_sd, tmp_path):
        out = tmp_path / "test.wav"
        result = record(output_path=out, sample_rate=16000, duration=0.1)
        assert result == str(out)
        assert os.path.exists(result)

    def test_creates_temp_file_when_no_output_path(self, _mock_sd):
        result = record(sample_rate=16000, duration=0.1)
        assert result is not None
        assert result.endswith(".wav")
        assert os.path.exists(result)
        os.unlink(result)  # cleanup

    def test_writes_valid_audio_data(self, _mock_sd, tmp_path):
        import soundfile as sf

        out = tmp_path / "test.wav"
        record(output_path=out, sample_rate=16000, duration=0.1)

        data, sr = sf.read(str(out))
        assert sr == 16000
        assert len(data) > 0

    def test_returns_none_when_no_frames(self, tmp_path):
        """If no audio frames are captured, record() returns None."""

        class EmptyInputStream:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_sd = MagicMock()
        mock_sd.InputStream = EmptyInputStream
        mock_sd.sleep = MagicMock(side_effect=lambda ms: None)

        out = tmp_path / "test.wav"
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = record(output_path=out, sample_rate=16000, duration=0.1)

        assert result is None

    def test_duration_mode_uses_sleep(self, _mock_sd, tmp_path):
        """With duration set, record() should call sd.sleep(duration * 1000)."""
        calls = []
        original_sleep = _mock_sd.sleep

        def tracking_sleep(ms):
            calls.append(ms)
            return original_sleep(ms)

        _mock_sd.sleep = tracking_sleep
        out = tmp_path / "test.wav"
        record(output_path=out, sample_rate=16000, duration=2.5)
        # First sleep call should be duration * 1000
        assert calls[0] == 2500

    def test_restores_sigint_handler(self, _mock_sd, tmp_path):
        original = signal.getsignal(signal.SIGINT)
        out = tmp_path / "test.wav"
        record(output_path=out, sample_rate=16000, duration=0.1)
        restored = signal.getsignal(signal.SIGINT)
        assert restored == original
