"""Record audio from a microphone.

Writes a WAV file to disk, prints the path to stdout.
Composable: mn-record | xargs mn-transcribe

Requires: pip install sounddevice soundfile
"""

import signal
import sys
import tempfile
from pathlib import Path


def record(output_path=None, sample_rate=16000, channels=1, duration=None):
    """Record audio from default microphone → WAV file path.

    Records until Ctrl-C (or duration seconds if set).
    Returns the path to the written WAV file.
    """
    import sounddevice as sd
    import soundfile as sf

    if output_path is None:
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = fd.name
        fd.close()

    output_path = str(output_path)
    frames = []
    stop = False

    def _callback(indata, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    def _stop(sig, frame):
        nonlocal stop
        stop = True

    prev_handler = signal.signal(signal.SIGINT, _stop)

    print(f"Recording → {output_path}  (Ctrl-C to stop)", file=sys.stderr)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels,
                            callback=_callback):
            if duration:
                sd.sleep(int(duration * 1000))
            else:
                while not stop:
                    sd.sleep(100)
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if not frames:
        print("No audio captured.", file=sys.stderr)
        return None

    import numpy as np
    audio = np.concatenate(frames, axis=0)
    sf.write(output_path, audio, sample_rate)
    seconds = len(audio) / sample_rate
    m, s = divmod(int(seconds), 60)
    print(f"Saved {m}:{s:02d} of audio.", file=sys.stderr)
    return output_path
