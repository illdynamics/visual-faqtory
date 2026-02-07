#!/usr/bin/env python3
"""
audio_reactive.py - Live Audio-Reactive Controller
═══════════════════════════════════════════════════════════════════════════════

Provides a simple, low-latency audio capture interface for the TURBO engine.

This controller captures audio from a microphone or loopback device using the
`sounddevice` library. It computes the root-mean-square (RMS) energy of the
incoming signal and performs a rudimentary peak/beat detection by comparing
the smoothed RMS against a user-defined threshold. The results are exposed
through a thread-safe `get_audio_state()` method so that callers can
periodically poll without blocking frame generation.

Key features:

  • Runs in the background using a non-blocking audio stream. The callback
    accumulates RMS values and updates shared state under a lock.

  • Gracefully handles missing dependencies or invalid devices. If
    initialisation fails, the controller disables itself and returns static
    zero-values to callers.

  • Configurable sample rate, block size, smoothing constant and beat
    threshold. These parameters may be tuned via the `config.yaml` under the
    `turbo.audio_reactive` section.

Returned state:
```
{
  "rms": float,   # 0.0–1.0 normalised RMS energy
  "beat": bool    # True if threshold crossing detected since last call
}
```

Part of QonQrete Visual FaQtory v0.3.5-beta
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Dict

try:
    import numpy as np  # type: ignore
except Exception as _e:
    np = None  # numpy is optional; used for RMS computation

logger = logging.getLogger(__name__)


class AudioReactiveController:
    """Capture live audio and compute smoothed RMS and simple beat detection.

    The controller starts a background audio stream on construction if
    `enabled` is True in the provided config. Otherwise, the controller
    remains inert. Audio capture uses the `sounddevice.InputStream` class.
    """

    def __init__(self, config: Dict[str, object]) -> None:
        # Configuration
        self.enabled: bool = bool(config.get("enabled", False))
        self.device: Optional[object] = config.get("device")  # type: ignore
        self.sample_rate: int = int(config.get("sample_rate", 44100) or 44100)
        self.block_size: int = int(config.get("block_size", 1024) or 1024)
        # Exponential moving average smoothing constant (0–1). Higher values
        # favour recent samples.
        self.rms_smooth: float = float(config.get("rms_smooth", 0.25) or 0.25)
        # Beat threshold on the smoothed RMS (0–1). When the smoothed RMS
        # crosses this value from below to above, a beat is signalled.
        self.beat_threshold: float = float(config.get("beat_threshold", 0.6) or 0.6)

        # Internal state
        self._current_rms: float = 0.0
        self._smoothed_rms: float = 0.0
        self._prev_above: bool = False
        self._beat: bool = False
        self._lock = threading.Lock()
        self._stream: Optional[object] = None
        self._running: bool = False
        self._error: bool = False

        # Start if enabled
        if self.enabled:
            self.start()

    def start(self) -> None:
        """Attempt to start the audio input stream.

        If `sounddevice` or `numpy` are unavailable or if the selected
        device cannot be opened, the controller disables itself gracefully.
        """
        if self._running or self._error:
            return
        # Check that numpy and sounddevice are available
        if np is None:
            logger.warning("[AudioReactive] numpy not available; disabling audio reactivity")
            self._error = True
            return
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            logger.warning(f"[AudioReactive] sounddevice import failed: {e}; disabling audio reactivity")
            self._error = True
            return
        try:
            # Define callback to process audio blocks
            def callback(indata, frames, time_info, status):  # type: ignore
                # Copy to avoid referencing memory that may be reused
                try:
                    # indata shape: (frames, channels). Flatten to mono by averaging if multi-channel.
                    data = indata
                    if data.ndim > 1:
                        # Average across channels
                        data = data.mean(axis=1)
                    # Compute RMS using numpy
                    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
                    # Normalise RMS roughly to 0–1 range; typical 16-bit PCM values lie within [-1, 1]
                    rms = max(0.0, min(1.0, rms))
                    with self._lock:
                        # Exponential moving average smoothing
                        alpha = max(0.0, min(1.0, self.rms_smooth))
                        self._smoothed_rms = (1.0 - alpha) * self._smoothed_rms + alpha * rms
                        # Beat detection: threshold crossing
                        above = self._smoothed_rms > self.beat_threshold
                        self._beat = above and not self._prev_above
                        self._prev_above = above
                        self._current_rms = self._smoothed_rms
                except Exception:
                    # Ensure no exceptions escape the callback
                    pass

            # Construct the input stream; default to mono (1 channel)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=1,
                callback=callback,
            )
            self._stream.start()
            self._running = True
            logger.info("[AudioReactive] Audio input stream started")
        except Exception as e:
            logger.warning(f"[AudioReactive] Failed to start audio stream: {e}; disabling audio reactivity")
            self._error = True
            try:
                if self._stream:
                    self._stream.close()
            finally:
                self._stream = None

    def stop(self) -> None:
        """Stop the audio input stream and reset internal state."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._running = False
        with self._lock:
            self._current_rms = 0.0
            self._smoothed_rms = 0.0
            self._prev_above = False
            self._beat = False
        logger.info("[AudioReactive] Audio input stream stopped")

    def is_running(self) -> bool:
        """Return True if the audio stream is active and not in error."""
        return self._running and not self._error

    def get_audio_state(self) -> Dict[str, object]:
        """Return the latest RMS and beat state.

        The returned RMS value is the smoothed RMS, clamped between 0 and 1.
        The beat flag resets automatically upon the next call when the
        threshold crossing condition is re-evaluated in the callback.
        """
        # Even if controller is disabled, always return a state dict.
        if not self.is_running():
            return {"rms": 0.0, "beat": False}
        with self._lock:
            # Copy values to avoid race conditions
            rms = float(self._current_rms)
            beat = bool(self._beat)
            # Reset beat flag; next callback will update
            self._beat = False
        return {"rms": rms, "beat": beat}
