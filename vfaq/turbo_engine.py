#!/usr/bin/env python3
"""
turbo_engine.py - TURBO Live Frame Generation Engine
═══════════════════════════════════════════════════════════════════════════════

Low‑latency loop for real‑time AI visual generation. This engine sits atop
ComfyUI and orchestrates prompt composition, seed advancement, macro
modulation and frame pacing. Version **v0.3.5-beta** adds the unified macro
with configurable momentary/toggle note modes, CC smoothing and a rich
command‑line interface, and retains the TouchDesigner integration and OSC
output introduced in v0.3.2‑beta. The audio‑reactive control layer from
v0.3.0‑beta remains, allowing RMS and beat detection to drive CFG and seed
drift. The sidecar triggers macros and continuous parameters via file
operations. Optional OSC broadcasting allows external systems (e.g.
TouchDesigner) to react to the current macro, crowd state and energy
without polling files. Both MIDI and audio reactivity may be toggled on the
fly via file triggers, and will defer to crowd prompts and manual macro
triggers when active.

Key capabilities:

  - SDXL Turbo / LCM single-frame generation via ComfyUI
  - Hot-reload tasq.md for instant prompt edits
  - Audio-reactive modulation (optional, hot-toggleable)
  - Crowd prompt integration with priority override
  - OBS overlay writing (current_frame.jpg + text files)
  - DJ macro file triggers

Bypasses InstruQtor/InspeQtor completely for minimal latency.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import os
import sys
import time
import json
import signal
import hashlib
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Live audio controller
from .audio_reactive import AudioReactiveController
# Long-run color stability (v0.3.5-beta)
from .color_stability import create_stability_controller
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptState:
    """Current prompt composition state."""
    base_prompt: str = ""
    macro: str = "CHILL"  # DROP | BUILD | CHILL
    crowd_prompt: str = ""
    crowd_name: str = ""
    crowd_expiry: float = 0.0  # unix timestamp
    next_crowd_preview: str = ""
    last_applied: str = ""
    seed: int = 42

    @property
    def crowd_active(self) -> bool:
        return bool(self.crowd_prompt and time.time() < self.crowd_expiry)


class TurboEngine:
    """
    Real-time frame generation engine.

    Usage:
        engine = TurboEngine(config, worqspace_dir)
        engine.run_live()  # blocks until stopped
    """

    def __init__(
        self,
        config: Dict[str, Any],
        worqspace_dir: Path,
        crowd_queue=None,
    ):
        self.config = config
        self.worqspace_dir = Path(worqspace_dir)
        self.turbo_cfg = config.get('turbo', {})
        self.crowd_cfg = config.get('crowd', {})

        # Turbo params
        self.steps = self.turbo_cfg.get('steps', 2)
        self.cfg_scale = self.turbo_cfg.get('cfg', 1.5)
        self.width = self.turbo_cfg.get('width', 768)
        self.height = self.turbo_cfg.get('height', 432)
        self.fps_target = self.turbo_cfg.get('fps_target', 12)
        self.seed_mode = self.turbo_cfg.get('seed_mode', 'drift')
        self.seed_drift = self.turbo_cfg.get('seed_drift', 3)
        self.negative_prompt = self.turbo_cfg.get('negative_prompt', '')

        # Output
        out_path = self.turbo_cfg.get('output_path', 'live_output/current_frame.jpg')
        self.output_dir = Path(out_path).parent
        self.output_file = Path(out_path).name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Backend
        self._backend = None
        self._workflow_template = None

        # Prompt state
        self.state = PromptState()
        self._load_base_prompt()

        # Crowd queue (direct reference, not HTTP)
        self.crowd_queue = crowd_queue
        self._crowd_poll_ms = self.crowd_cfg.get('integration', {}).get('poll_ms', 250)
        self._crowd_mode = self.crowd_cfg.get('integration', {}).get('mode', 'blend')
        self._crowd_duration = self.crowd_cfg.get('integration', {}).get(
            'blend', {}
        ).get('duration_seconds', 20)

        # Overlay writer
        from .overlay_writer import OverlayWriter
        overlay_cfg = self.crowd_cfg.get('overlays', {})
        self.overlay = OverlayWriter(
            out_dir=overlay_cfg.get('out_dir', str(self.output_dir)),
            config=overlay_cfg,
        )

        # Hot reload
        hr_cfg = self.turbo_cfg.get('hot_reload', {})
        self._hot_reload = hr_cfg.get('enabled', True)
        self._hot_reload_ms = hr_cfg.get('poll_ms', 200)
        self._tasq_mtime = 0.0

        # Macro triggers
        self._macro_dir = self.output_dir

        # Stats
        self.frames_generated = 0
        self.total_gen_ms = 0.0
        self._running = False

        # ─── Audio Reactive Control (v0.3.5-beta) ───────────────────────────
        # Configuration sections for audio reactivity, live toggling and priority
        self._audio_cfg: Dict[str, Any] = self.turbo_cfg.get('audio_reactive', {})
        self._live_toggle_cfg: Dict[str, Any] = self.turbo_cfg.get('live_toggle', {})
        self._priority_cfg: Dict[str, Any] = self.turbo_cfg.get('priority', {})
        # Preserve base CFG and seed drift so that audio modulation can revert
        self._base_cfg_scale: float = float(self.cfg_scale)
        self._base_seed_drift: int = int(self.seed_drift)
        # Audio controller instance and enable flag
        self.audio_controller: Optional[AudioReactiveController] = None
        self.audio_enabled: bool = False
        # Manual macro override. When set to a macro string (e.g. 'DROP' or 'BUILD')
        # audio beat triggers will not modify the macro. CHILL clears override.
        self.manual_macro_override: Optional[str] = None

        # Continuous macro factors from MIDI sidecar
        # Values read from macro_INTENSITY and macro_ENERGY files (0–1)
        self._intensity_value: float = 0.0
        self._energy_value: float = 0.0

        # ─── Background Polling Thread (v0.3.5-beta) ────────────────────
        # The control layer is implemented as logical components inside
        # TurboEngine, not as separate classes. Three components:
        #   1. Audio influence  — RMS/beat → CFG/seed drift/macro
        #   2. Macro influence  — file-based momentary/toggle/value macros
        #   3. Crowd override   — when crowd is active, audio is paused
        #
        # A background daemon thread polls macro files and audio toggle
        # state every 100-200ms, independent of the render loop. The
        # render loop reads only cached values (non-blocking). This
        # guarantees ≤200ms response to macro_AUDIO toggle even when
        # frame generation takes seconds.
        self._poll_interval_ms: int = int(
            self._live_toggle_cfg.get('poll_interval_ms', 150)
        )
        self._poll_lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        # Cached state (written by poll thread, read by render loop)
        self._cached_audio_enabled: bool = False
        self._cached_macro: str = 'CHILL'
        self._cached_manual_override: Optional[str] = None
        self._cached_intensity: float = 0.0
        self._cached_energy: float = 0.0

        # ─── OSC Output (v0.3.5-beta) ───────────────────────────────────────
        # Load OSC configuration (optional)
        self._osc_cfg: Dict[str, Any] = config.get('osc', {}) if isinstance(config, dict) else {}
        self.osc_client: Optional[object] = None  # type: ignore
        self._osc_send_ms: int = 100
        self._last_osc_ts: float = 0.0
        if self._osc_cfg.get('enabled'):
            try:
                from .osc_out import OSCClient  # Late import to avoid dependency if disabled
                host = self._osc_cfg.get('host', '127.0.0.1')
                port = self._osc_cfg.get('port', 6000)
                address = self._osc_cfg.get('address', '/visual_faqtory')
                self.osc_client = OSCClient(host=host, port=int(port), address=str(address))
                self._osc_send_ms = int(self._osc_cfg.get('send_ms', 100))
                self._last_osc_ts = 0.0
                if not self.osc_client.is_active():
                    # If client failed to initialise, disable OSC
                    self.osc_client = None
            except Exception as e:
                logger.warning(f"[TURBO] OSC initialisation failed: {e}")
                self.osc_client = None

        # Signal handler
        self._setup_signals()

        # ─── Color Stability Controller (v0.3.5-beta) ────────────────────
        self.stability = create_stability_controller(config)
        if self.stability:
            logger.info("[TURBO] Color stability controller enabled")

    def _setup_signals(self):
        def handler(signum, frame):
            logger.info("\n[TURBO] Shutdown requested...")
            self._running = False
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    # ──────────────────────────────────────────────────────────────────────────
    # Audio Reactive Helpers
    #
    def _resolve_toggle_file(self) -> Path:
        """Compute the full path to the live toggle file.

        The toggle file may be specified as an absolute path or relative to
        the TURBO output directory. If the path is absolute it is returned
        unchanged; otherwise, it is joined with `self.output_dir`.
        """
        toggle_name = self._live_toggle_cfg.get('toggle_file', 'macro_AUDIO')
        toggle_path = Path(toggle_name)
        if toggle_path.is_absolute():
            return toggle_path
        return self.output_dir / toggle_path.name

    def _apply_continuous_macros(self) -> None:
        """Apply continuous macro adjustments from macro_INTENSITY/ENERGY files.

        The MIDI sidecar can write floating-point values (0–1) to
        `macro_INTENSITY` and `macro_ENERGY`. These values modulate the
        current `cfg_scale` and `seed_drift` multiplicatively. When the
        files are absent, the corresponding factors revert to 0 and no
        scaling is applied. Errors are ignored to ensure frame safety.
        """
        # New implementation for continuous macros (v0.3.5-beta)
        try:
            # Determine paths for continuous macro files
            intensity_path = self.output_dir / "macro_INTENSITY"
            energy_path = self.output_dir / "macro_ENERGY"
            intensity_val: Optional[float] = None
            energy_val: Optional[float] = None
            # Read intensity value (normalised 0–1)
            if intensity_path.exists():
                try:
                    raw = intensity_path.read_text(encoding="utf-8").strip()
                    intensity_val = max(0.0, min(1.0, float(raw)))
                except Exception:
                    intensity_val = None
            # Read energy value (normalised 0–1)
            if energy_path.exists():
                try:
                    raw = energy_path.read_text(encoding="utf-8").strip()
                    energy_val = max(0.0, min(1.0, float(raw)))
                except Exception:
                    energy_val = None
            # Revert previous intensity scaling
            prev_intensity = getattr(self, "_intensity_value", 0.0)
            if prev_intensity not in (None, 0.0):
                try:
                    self.cfg_scale = float(self.cfg_scale) / (1.0 + prev_intensity)
                except Exception:
                    pass
            # Revert previous energy scaling
            prev_energy = getattr(self, "_energy_value", 0.0)
            if prev_energy not in (None, 0.0):
                try:
                    base_drift = float(self.seed_drift) / (1.0 + prev_energy)
                    self.seed_drift = max(0, int(round(base_drift)))
                except Exception:
                    pass
            # Apply new intensity scaling
            if intensity_val is not None:
                self._intensity_value = intensity_val
                try:
                    self.cfg_scale = float(self.cfg_scale) * (1.0 + intensity_val)
                except Exception:
                    pass
            else:
                self._intensity_value = 0.0
            # Apply new energy scaling
            if energy_val is not None:
                self._energy_value = energy_val
                try:
                    sd = float(self.seed_drift) * (1.0 + energy_val)
                    self.seed_drift = max(0, int(round(sd)))
                except Exception:
                    pass
            else:
                self._energy_value = 0.0
        except Exception as e:
            # Log and ignore unexpected errors to keep frame loop safe
            logger.debug(f"[TURBO] Continuous macro error: {e}")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Background Polling Thread (v0.3.5-beta)
    #
    # Polls macro files and audio toggle state every 100-200ms, independent
    # of the render loop. The render loop reads only cached values.
    #
    def _start_poll_thread(self) -> None:
        """Start the background macro/audio polling daemon thread."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="turbo-macro-poll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(
            f"[TURBO] Background poll thread started "
            f"(interval={self._poll_interval_ms}ms)"
        )

    def _stop_poll_thread(self) -> None:
        """Stop the background polling thread gracefully."""
        # The thread checks self._running and exits when False.
        # Since it's a daemon, it will also die when the process exits.
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        """Background loop: poll macros and audio toggle state.

        Runs until self._running is False. Sleeps for _poll_interval_ms
        between iterations. Never blocks shutdown (daemon thread + running flag).
        """
        interval_sec = max(0.05, self._poll_interval_ms / 1000.0)
        while self._running:
            try:
                self._poll_macros_and_audio()
            except Exception as e:
                logger.debug(f"[TURBO] Poll thread error: {e}")
            time.sleep(interval_sec)

    def _poll_macros_and_audio(self) -> None:
        """Read macro files and audio toggle, update cached state under lock.

        This runs on the background thread. All file I/O happens here.
        The render loop only reads the cached values via _read_cached_state().
        """
        # ── Read momentary macros (DROP > BUILD > CHILL) ─────────────
        found_macro = None
        for macro_name in ['DROP', 'BUILD', 'CHILL']:
            trigger = self._macro_dir / f"macro_{macro_name}"
            if trigger.exists():
                found_macro = macro_name
                break

        new_macro = found_macro or 'CHILL'
        new_override = None
        if found_macro and found_macro != 'CHILL':
            new_override = found_macro

        # ── Read audio toggle (ARMED vs ACTIVE semantics) ─────────────
        # audio_reactive.enabled = ARMED (feature available)
        # macro_AUDIO file       = ACTIVE switch (only matters when armed)
        #
        # enabled=false → audio OFF, toggle file ignored
        # enabled=true  + live_toggle + file exists → ON
        # enabled=true  + live_toggle + file missing → OFF
        # enabled=true  + no live_toggle             → ON (armed = always active)
        armed = bool(self._audio_cfg.get('enabled', False))
        use_toggle = bool(self._live_toggle_cfg.get('enabled', False))
        if not armed:
            new_audio_enabled = False
            # Log once if toggle file exists while not armed
            if use_toggle:
                toggle_path = self._resolve_toggle_file()
                if toggle_path.exists() and not getattr(self, '_armed_warning_logged', False):
                    logger.warning(
                        "[TURBO] macro_AUDIO file exists but audio_reactive.enabled=false "
                        "(not armed). Toggle file is ignored. Set audio_reactive.enabled: "
                        "true in config to arm audio reactivity."
                    )
                    self._armed_warning_logged = True
                elif not toggle_path.exists():
                    self._armed_warning_logged = False  # Reset so we warn again if re-touched
        elif use_toggle:
            toggle_path = self._resolve_toggle_file()
            new_audio_enabled = toggle_path.exists()
        else:
            # Armed + no toggle mechanism = always active
            new_audio_enabled = True

        # ── Read continuous macros (INTENSITY, ENERGY) ───────────────
        new_intensity = 0.0
        new_energy = 0.0
        intensity_path = self._macro_dir / "macro_INTENSITY"
        energy_path = self._macro_dir / "macro_ENERGY"
        if intensity_path.exists():
            try:
                raw = intensity_path.read_text(encoding="utf-8").strip()
                new_intensity = max(0.0, min(1.0, float(raw)))
            except Exception:
                pass
        if energy_path.exists():
            try:
                raw = energy_path.read_text(encoding="utf-8").strip()
                new_energy = max(0.0, min(1.0, float(raw)))
            except Exception:
                pass

        # ── Write cached state under lock ────────────────────────────
        with self._poll_lock:
            prev_macro = self._cached_macro
            prev_audio = self._cached_audio_enabled

            self._cached_macro = new_macro
            self._cached_manual_override = new_override
            self._cached_audio_enabled = new_audio_enabled
            self._cached_intensity = new_intensity
            self._cached_energy = new_energy

        # Log state changes (outside lock)
        if new_macro != prev_macro:
            if new_macro == 'CHILL' and prev_macro != 'CHILL':
                logger.info("[TURBO] No macro file present → reverting to CHILL")
            else:
                logger.info(f"[TURBO] Macro active: {new_macro}")
        if new_audio_enabled != prev_audio:
            logger.info(f"[TURBO] Audio toggle: {'ON' if new_audio_enabled else 'OFF'}")

    def _read_cached_state(self) -> dict:
        """Read cached macro/audio state (called from render loop, non-blocking)."""
        with self._poll_lock:
            return {
                'macro': self._cached_macro,
                'manual_override': self._cached_manual_override,
                'audio_enabled': self._cached_audio_enabled,
                'intensity': self._cached_intensity,
                'energy': self._cached_energy,
            }

    def _apply_cached_audio_state(self, cached: dict) -> None:
        """Start/stop the audio controller based on cached toggle state.

        Called from the render loop after reading cached state. The actual
        file polling happened on the background thread already.
        """
        new_enabled = cached['audio_enabled']
        if new_enabled and not self.audio_enabled:
            self.audio_enabled = True
            if not self.audio_controller:
                self.audio_controller = AudioReactiveController(self._audio_cfg)
            elif not self.audio_controller.is_running():
                self.audio_controller.start()
        elif not new_enabled and self.audio_enabled:
            self.audio_enabled = False
            if self.audio_controller:
                self.audio_controller.stop()

    def _apply_continuous_macros_cached(self, cached: dict) -> None:
        """Apply INTENSITY/ENERGY scaling from cached background-thread values.

        Replaces the old _apply_continuous_macros() which did file I/O in
        the render loop.
        """
        intensity_val = cached['intensity']
        energy_val = cached['energy']

        # Revert previous intensity scaling
        prev_intensity = self._intensity_value
        if prev_intensity not in (None, 0.0):
            try:
                self.cfg_scale = float(self.cfg_scale) / (1.0 + prev_intensity)
            except Exception:
                pass

        # Revert previous energy scaling
        prev_energy = self._energy_value
        if prev_energy not in (None, 0.0):
            try:
                base_drift = float(self.seed_drift) / (1.0 + prev_energy)
                self.seed_drift = max(0, int(round(base_drift)))
            except Exception:
                pass

        # Apply new intensity
        if intensity_val > 0.0:
            self._intensity_value = intensity_val
            try:
                self.cfg_scale = float(self.cfg_scale) * (1.0 + intensity_val)
            except Exception:
                pass
        else:
            self._intensity_value = 0.0

        # Apply new energy
        if energy_val > 0.0:
            self._energy_value = energy_val
            try:
                sd = float(self.seed_drift) * (1.0 + energy_val)
                self.seed_drift = max(0, int(round(sd)))
            except Exception:
                pass
        else:
            self._energy_value = 0.0

    def _send_osc_update(self) -> None:
        """Send an OSC message with macro, crowd flag and energy if enabled.

        This method is non-blocking and throttled by the configured send
        interval. If OSC is not enabled or the client is inactive, it
        returns immediately.
        """
        client = getattr(self, 'osc_client', None)
        if not client:
            return
        try:
            # Check throttle interval
            now_ms = time.time() * 1000.0
            if now_ms - getattr(self, '_last_osc_ts', 0.0) < getattr(self, '_osc_send_ms', 100):
                return
            # Update timestamp
            self._last_osc_ts = now_ms
            # Compose payload and send
            macro = self.state.macro or ""
            crowd_active = bool(self.state.crowd_active)
            energy = max(self._intensity_value, self._energy_value)
            client.send(macro=macro, crowd_active=crowd_active, energy=float(energy))
        except Exception as e:
            # Never raise from OSC errors
            logger.debug(f"[TURBO] OSC send error: {e}")

    def _update_audio_enabled(self) -> None:
        """Enable or disable the audio controller based on ARMED + ACTIVE semantics.

        ARMED gate: ``turbo.audio_reactive.enabled`` must be ``true`` for the
        audio-reactive feature to be available at all. If not armed, the toggle
        file is ignored and audio stays OFF.

        ACTIVE switch: when armed *and* ``live_toggle.enabled: true``, the
        presence of the toggle file (default ``macro_AUDIO``) determines
        whether audio is currently active. When armed but live toggling is
        disabled, audio is always active.

        Starting/stopping the controller is handled gracefully; failures will
        disable the controller until the next re-evaluation.
        """
        armed = bool(self._audio_cfg.get('enabled', False))
        use_toggle = bool(self._live_toggle_cfg.get('enabled', False))

        if not armed:
            new_enabled = False
        elif use_toggle:
            toggle_path = self._resolve_toggle_file()
            new_enabled = toggle_path.exists()
        else:
            new_enabled = True  # armed + no toggle = always active

        # Apply changes
        if new_enabled and not self.audio_enabled:
            # Activate audio. Instantiate controller if absent or not running.
            self.audio_enabled = True
            if not self.audio_controller:
                self.audio_controller = AudioReactiveController(self._audio_cfg)
            else:
                # If controller exists but stopped, attempt restart
                if not self.audio_controller.is_running():
                    self.audio_controller.start()
        elif not new_enabled and self.audio_enabled:
            # Deactivate audio
            self.audio_enabled = False
            if self.audio_controller:
                self.audio_controller.stop()

    def _load_base_prompt(self):
        """Load base prompt from tasq.md."""
        tasq_path = self.worqspace_dir / self.turbo_cfg.get('prompt_source', 'tasq.md')
        if tasq_path.exists():
            text = tasq_path.read_text(encoding='utf-8')
            # Strip YAML frontmatter
            if text.startswith('---'):
                parts = text.split('---', 2)
                if len(parts) >= 3:
                    text = parts[2]
            self.state.base_prompt = ' '.join(text.strip().split())
            self._tasq_mtime = tasq_path.stat().st_mtime
            logger.info(f"[TURBO] Base prompt: '{self.state.base_prompt[:80]}...'")
        else:
            self.state.base_prompt = "abstract flowing colors, cinematic"
            logger.warning(f"[TURBO] No tasq.md found, using default prompt")

    def _hot_reload_check(self):
        """Check if tasq.md changed and reload."""
        if not self._hot_reload:
            return
        tasq_path = self.worqspace_dir / self.turbo_cfg.get('prompt_source', 'tasq.md')
        if tasq_path.exists():
            mtime = tasq_path.stat().st_mtime
            if mtime > self._tasq_mtime:
                self._load_base_prompt()
                logger.info("[TURBO] Hot-reloaded tasq.md")

    def _check_macro_triggers(self):
        """Check for DJ macro file triggers.

        v0.3.5-beta unified macro contract:
          - Momentary macros (DROP, BUILD, CHILL): active WHILE file exists.
            When the file is removed (e.g. MIDI note-off), the state reverts
            to CHILL. Files are NEVER auto-deleted by Turbo.
          - Toggle macros (AUDIO): active if file exists.
          - Value macros (INTENSITY, ENERGY): read float from file content.

        Priority: DROP > BUILD > CHILL (first found wins).
        If no momentary macro file exists, revert to CHILL.
        """
        # Check momentary macros in priority order
        found_macro = None
        for macro_name in ['DROP', 'BUILD', 'CHILL']:
            trigger = self._macro_dir / f"macro_{macro_name}"
            if trigger.exists():
                found_macro = macro_name
                break  # Highest priority wins

        if found_macro:
            if self.state.macro != found_macro:
                logger.info(f"[TURBO] Macro active: {found_macro}")
            self.state.macro = found_macro
            # Manual override: CHILL clears override, others enable it
            if found_macro == 'CHILL':
                self.manual_macro_override = None
            else:
                self.manual_macro_override = found_macro
        else:
            # No momentary macro file present → revert to CHILL
            if self.state.macro != 'CHILL':
                logger.info("[TURBO] No macro file present → reverting to CHILL")
            self.state.macro = 'CHILL'
            self.manual_macro_override = None

    def _poll_crowd_queue(self):
        """Non-blocking crowd queue poll."""
        if not self.crowd_queue:
            return

        # Check if current crowd prompt expired
        if self.state.crowd_active:
            return  # Still active, don't pull yet

        # Pop next
        item = self.crowd_queue.pop_next()
        if item:
            self.state.crowd_prompt = item.prompt
            self.state.crowd_name = item.name
            self.state.crowd_expiry = time.time() + self._crowd_duration
            logger.info(f"[TURBO] Crowd prompt active: @{item.name}: '{item.prompt[:50]}'")

            # Toast
            self.overlay.write_toast(item.name, item.prompt)

        # Update next preview
        peek = self.crowd_queue.peek_next()
        self.state.next_crowd_preview = (
            f"@{peek.name}: {peek.prompt[:40]}" if peek else ""
        )

    def _compose_prompt(self) -> str:
        """Compose effective prompt from all sources."""
        parts = [self.state.base_prompt]

        # Macro modifier
        macro_map = {
            'DROP': ', intense energy, explosive motion, high contrast, bass heavy',
            'BUILD': ', rising tension, building energy, bright highlights, anticipation',
            'CHILL': '',
        }
        macro_suffix = macro_map.get(self.state.macro, '')
        if macro_suffix:
            parts.append(macro_suffix)

        # Crowd integration
        if self.state.crowd_active:
            if self._crowd_mode == 'takeover':
                return self.state.crowd_prompt
            elif self._crowd_mode == 'blend':
                parts.append(f", crowd variation: {self.state.crowd_prompt}")

        prompt = ''.join(parts)
        self.state.last_applied = prompt
        return prompt

    def _advance_seed(self):
        """Advance seed based on mode."""
        if self.state.macro == 'DROP':
            self.state.seed += self.seed_drift * 10
        elif self.seed_mode == 'drift':
            self.state.seed += self.seed_drift
        elif self.seed_mode == 'beatjump':
            self.state.seed += 13

    def _init_backend(self):
        """Initialize ComfyUI backend for turbo generation."""
        from .backends import ComfyUIBackend
        backend_type = self.turbo_cfg.get('backend', 'comfyui')

        if backend_type == 'comfyui':
            api_url = self.config.get('comfyui', {}).get(
                'api_url', self.config.get('backend', {}).get('api_url', 'http://localhost:8188')
            )
            self._backend = ComfyUIBackend({'api_url': api_url})

            # Load workflow
            mode = self.turbo_cfg.get('mode', 'sdxl_turbo')
            wf_map = {
                'sdxl_turbo': 'worqspace/workflows/turbo_sdxl.json',
                'lcm': 'worqspace/workflows/turbo_lcm.json',
            }
            wf_path = Path(wf_map.get(mode, wf_map['sdxl_turbo']))
            if wf_path.exists():
                self._workflow_template = json.loads(wf_path.read_text())
                self._workflow_template.pop('_meta', None)
                logger.info(f"[TURBO] Loaded workflow: {wf_path}")
            else:
                logger.warning(f"[TURBO] Workflow not found: {wf_path}, will use default")
                self._workflow_template = self._build_default_workflow()
        else:
            logger.error(f"[TURBO] Unsupported backend: {backend_type}")
            raise RuntimeError(f"Turbo requires comfyui backend, got: {backend_type}")

    def _build_default_workflow(self) -> Dict:
        """Build a default SDXL Turbo workflow."""
        ckpt = self.config.get('comfyui', {}).get('sdxl_turbo_ckpt', 'sd_xl_turbo_1.0_fp16.safetensors')
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "POSITIVE", "clip": ["1", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "NEGATIVE", "clip": ["1", 1]}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": self.width, "height": self.height, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42, "steps": self.steps, "cfg": self.cfg_scale,
                    "sampler_name": "euler_ancestral", "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0], "positive": ["2", 0],
                    "negative": ["3", 0], "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "turbo_live", "images": ["6", 0]}
            }
        }

    def _generate_frame(self, prompt: str, negative: str, seed: int) -> Optional[bytes]:
        """Generate a single frame via ComfyUI."""
        import copy
        try:
            import requests
        except ImportError:
            logger.error("[TURBO] requests not installed")
            return None

        workflow = copy.deepcopy(self._workflow_template)

        # Graph-based injection
        workflow = self._backend._inject_prompts_graph_based(workflow, prompt, negative or "low quality")

        # Inject KSampler params
        for nid, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['seed'] = seed
                node['inputs']['steps'] = self.steps
                node['inputs']['cfg'] = self.cfg_scale
            elif node.get('class_type') == 'EmptyLatentImage':
                node['inputs']['width'] = self.width
                node['inputs']['height'] = self.height

        # Queue
        client_id = hashlib.sha256(f"turbo_{time.time()}".encode()).hexdigest()[:16]
        try:
            resp = requests.post(
                f"{self._backend.api_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=10
            )
            if resp.status_code != 200:
                logger.warning(f"[TURBO] Queue failed: {resp.status_code}")
                return None
            prompt_id = resp.json().get('prompt_id')
        except Exception as e:
            logger.warning(f"[TURBO] Queue error: {e}")
            return None

        # Poll for result
        start = time.time()
        timeout = self.turbo_cfg.get('timeout', 30)
        while time.time() - start < timeout:
            try:
                hist = requests.get(f"{self._backend.api_url}/history/{prompt_id}", timeout=5).json()
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get('status', {}).get('status_str', '')
                    if status == 'error':
                        logger.warning("[TURBO] ComfyUI execution error")
                        return None

                    for node_id, output in entry.get('outputs', {}).items():
                        if 'images' in output:
                            for img in output['images']:
                                from urllib.parse import urlencode
                                params = urlencode({
                                    'filename': img['filename'],
                                    'subfolder': img.get('subfolder', ''),
                                    'type': img.get('type', 'output')
                                })
                                dl = requests.get(
                                    f"{self._backend.api_url}/view?{params}", timeout=10
                                )
                                if dl.status_code == 200 and len(dl.content) > 100:
                                    return dl.content
                    return None
            except Exception:
                pass
            time.sleep(0.1)

        logger.warning("[TURBO] Frame generation timed out")
        return None

    def _update_overlays(self):
        """Update OBS overlay files."""
        crowd_status = ""
        if self.state.crowd_active:
            remain = self.state.crowd_expiry - time.time()
            crowd_status = f"@{self.state.crowd_name}: '{self.state.crowd_prompt[:30]}' ({remain:.0f}s)"

        self.overlay.write_now(
            prompt=self.state.base_prompt[:60],
            macro=self.state.macro if self.state.macro != 'CHILL' else '',
            crowd_status=crowd_status,
        )
        self.overlay.write_next(self.state.next_crowd_preview)

        if self.crowd_queue:
            items = [i.to_dict() for i in self.crowd_queue.list_top(5)]
            self.overlay.write_queue(items)

    def run_live(self):
        """Main turbo generation loop. Blocks until stopped.

        Control Layer Architecture (v0.3.5-beta):
          The control layer is implemented as logical components inside
          TurboEngine, not as separate classes:

            1. Audio influence  — RMS/beat detection → CFG/seed drift/macro
            2. Macro influence  — file-based momentary/toggle/value macros
            3. Crowd override   — when crowd is active, audio influence = 0

          A background daemon thread polls macro files and audio toggle
          state every 100-200ms, independent of the frame render loop.
          The render loop reads only cached values via _read_cached_state().
          This guarantees ≤200ms response to macro_AUDIO toggle even when
          individual frame generation takes multiple seconds.
        """
        logger.info("=" * 60)
        logger.info("QonQrete Visual FaQtory v0.3.5-beta — TURBO LIVE MODE")
        logger.info("=" * 60)
        logger.info(f"  Resolution: {self.width}×{self.height}")
        logger.info(f"  Steps: {self.steps} | CFG: {self.cfg_scale}")
        logger.info(f"  FPS target: {self.fps_target}")
        logger.info(f"  Output: {self.output_dir / self.output_file}")
        logger.info(f"  Crowd: {'enabled' if self.crowd_queue else 'disabled'}")
        logger.info(f"  Poll thread: {self._poll_interval_ms}ms interval")
        logger.info("=" * 60)

        self._init_backend()
        self._running = True

        # Start the background macro/audio polling thread (v0.3.5-beta)
        self._start_poll_thread()

        target_interval = 1.0 / max(1, self.fps_target)
        last_hot_reload = 0.0
        last_crowd_poll = 0.0
        last_frame_bytes = None

        while self._running:
            loop_start = time.time()

            # Hot reload check (prompt file only — macros handled by poll thread)
            if loop_start - last_hot_reload > self._hot_reload_ms / 1000:
                self._hot_reload_check()
                last_hot_reload = loop_start

            # Crowd poll
            if self.crowd_queue and loop_start - last_crowd_poll > self._crowd_poll_ms / 1000:
                try:
                    self._poll_crowd_queue()
                except Exception as e:
                    logger.debug(f"[TURBO] Crowd poll error: {e}")
                last_crowd_poll = loop_start

            # ── Read cached state from background poll thread (non-blocking) ──
            cached = self._read_cached_state()

            # Apply macro state from cached values
            self.state.macro = cached['macro']
            self.manual_macro_override = cached['manual_override']

            # Apply audio toggle from cached values (start/stop controller)
            try:
                self._apply_cached_audio_state(cached)
            except Exception as e:
                logger.debug(f"[TURBO] Audio state apply error: {e}")

            # Audio reactive update (CFG/seed drift/macro) — non-blocking
            try:
                crowd_override = bool(
                    self._priority_cfg.get('crowd_overrides_audio', True)
                    and self.state.crowd_active
                )
                if self.audio_enabled and not crowd_override:
                    audio_state = (
                        self.audio_controller.get_audio_state()
                        if self.audio_controller
                        else {"rms": 0.0, "beat": False}
                    )
                    rms_val = max(0.0, min(1.0, float(audio_state.get('rms', 0.0))))
                    mappings = self._audio_cfg.get('mappings', {})
                    if 'rms_to_cfg' in mappings:
                        try:
                            cfg_min, cfg_max = mappings['rms_to_cfg']
                            self.cfg_scale = float(cfg_min) + (float(cfg_max) - float(cfg_min)) * rms_val
                        except Exception:
                            self.cfg_scale = self._base_cfg_scale
                    if 'rms_to_seed_drift' in mappings:
                        try:
                            sd_min, sd_max = mappings['rms_to_seed_drift']
                            new_drift = float(sd_min) + (float(sd_max) - float(sd_min)) * rms_val
                            self.seed_drift = int(round(new_drift))
                        except Exception:
                            self.seed_drift = self._base_seed_drift
                    if audio_state.get('beat', False) and not self.manual_macro_override:
                        macro_name = mappings.get('beat_macro', 'DROP')
                        if isinstance(macro_name, str) and macro_name:
                            self.state.macro = macro_name
                else:
                    if self.audio_enabled and crowd_override:
                        logger.debug("[TURBO] Audio paused: crowd override active")
                    self.cfg_scale = self._base_cfg_scale
                    self.seed_drift = self._base_seed_drift
            except Exception as e:
                logger.debug(f"[TURBO] Audio update error: {e}")

            # Continuous macro scaling from cached values (non-blocking)
            try:
                self._apply_continuous_macros_cached(cached)
            except Exception as e:
                logger.debug(f"[TURBO] Continuous macro apply error: {e}")

            # Compose prompt
            prompt = self._compose_prompt()
            self._advance_seed()

            # Generate frame
            gen_start = time.time()
            frame_bytes = self._generate_frame(
                prompt, self.negative_prompt, self.state.seed
            )
            gen_ms = (time.time() - gen_start) * 1000

            if frame_bytes:
                # Apply color stability correction (v0.3.5-beta)
                if self.stability:
                    # Use a 10ms time budget in live mode — skip if too slow
                    frame_bytes = self.stability.process_frame_bytes(
                        frame_bytes, time_budget_ms=10.0
                    )
                    # Apply generation modifiers from stability controller
                    mods = self.stability.get_generation_modifiers()
                    if mods.get("collapse_active"):
                        self.cfg_scale = max(1.0, self._base_cfg_scale - mods.get("cfg_reduce", 0))
                        self.seed_drift = max(0, int(
                            self._base_seed_drift * (1.0 - mods.get("seed_drift_reduce", 0))
                        ))

                last_frame_bytes = frame_bytes
                self.overlay.write_frame_jpg(frame_bytes, self.output_file)
                self.frames_generated += 1
                self.total_gen_ms += gen_ms

                if self.frames_generated % 10 == 0:
                    avg_ms = self.total_gen_ms / self.frames_generated
                    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                    queue_depth = self.crowd_queue.depth() if self.crowd_queue else 0
                    logger.info(
                        f"[TURBO] Frame {self.frames_generated}: "
                        f"{gen_ms:.0f}ms (avg {avg_ms:.0f}ms, ~{fps:.1f}fps) "
                        f"| queue={queue_depth} | macro={self.state.macro}"
                    )
            else:
                # Keep last good frame visible
                if last_frame_bytes:
                    logger.debug("[TURBO] Using last good frame")

            # Update overlays
            self._update_overlays()

            # OSC output (non-blocking)
            try:
                self._send_osc_update()
            except Exception:
                pass

            # Frame pacing
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        self._stop_poll_thread()
        logger.info(f"[TURBO] Stopped after {self.frames_generated} frames")
        avg_ms = self.total_gen_ms / max(1, self.frames_generated)
        logger.info(f"[TURBO] Average: {avg_ms:.0f}ms/frame")

    def stats(self) -> Dict[str, Any]:
        """Return engine stats."""
        avg_ms = self.total_gen_ms / max(1, self.frames_generated)
        return {
            'frames_generated': self.frames_generated,
            'avg_ms_per_frame': round(avg_ms, 1),
            'fps_effective': round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
            'macro': self.state.macro,
            'crowd_active': self.state.crowd_active,
            'seed': self.state.seed,
        }


__all__ = ['TurboEngine', 'PromptState']
