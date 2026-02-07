#!/usr/bin/env python3
"""
midi_sidecar.py – MIDI→macro bridge for Visual FaQtory
═══════════════════════════════════════════════════════════════════════════════

This standalone utility listens to a hardware MIDI input and translates
note/CC events into file‑based macros consumed by the Visual FaQtory TURBO
engine. It runs as a separate process and never imports any GPU, ComfyUI or
Turbo modules. Instead it creates, touches or removes files in a specified
directory (``live_output/`` by default). Turbo polls these files on each
frame and applies macros and continuous parameters without blocking or
introducing latency.

Key Features
------------

  • **Port discovery and selection**: Automatically lists available MIDI
    input ports and selects one by exact name or substring. Fallback to
    the only available port if unambiguous. Robust reconnection logic
    attempts to reopen the device if it is disconnected during a run.

  • **Note mapping with modes**: Each note number can be mapped to an
    action (DROP, BUILD, CHILL, AUDIO) with a *mode*: ``momentary`` creates
    a macro file on note on and removes it on note off; ``toggle`` flips
    the macro file’s presence on note on and ignores note off.

  • **Control change mapping with smoothing**: CC values are normalised
    between configurable ``min`` and ``max`` bounds and optionally smoothed
    using an exponential moving average. The result is written atomically
    to ``macro_INTENSITY`` or ``macro_ENERGY`` files as a floating‑point
    number.

  • **CLI overrides**: Most configuration parameters can be overridden at
    the command line, including port selection, output directory, logging
    level, dry‑run mode, note off cleanup, default note mode, CC verbosity
    and a self‑test for file I/O helpers.

  • **Gig‑safe isolation**: The sidecar never touches the render pipeline.
    If it crashes or no MIDI device is present, Turbo continues unaffected.

Usage:

    python sidecar/midi_sidecar.py --list
    python sidecar/midi_sidecar.py --name "Arduino" --out-dir live_output
    python sidecar/midi_sidecar.py --port "USB MIDI Interface" --dry-run

See the documentation (DOCUMENTATION.md §14) for a full description of
available options and the configuration schema.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML is optional; config may be empty

try:
    import mido  # type: ignore
except Exception:
    mido = None


# -----------------------------------------------------------------------------
# Helper functions for file I/O
# -----------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    """Write a string to a file atomically by writing to a temporary file
    and renaming it. Ensures readers never see a partially written file.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def touch_file(path: Path) -> None:
    """Create a file if it does not exist or update its modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        os.utime(path, None)


def remove_file(path: Path) -> None:
    """Remove a file if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def run_self_test() -> int:
    """Verify that touch/remove/atomic_write helpers work as expected.
    Creates a temporary directory, writes test files and checks their contents.
    Returns 0 on success and 1 on failure.
    """
    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        # Test touch
        f1 = base / "touch_test"
        try:
            touch_file(f1)
            if not f1.exists():
                print("[SELF-TEST] touch_file did not create file", file=sys.stderr)
                ok = False
        except Exception as e:
            print(f"[SELF-TEST] touch_file raised error: {e}", file=sys.stderr)
            ok = False
        # Test atomic write
        f2 = base / "atomic_test"
        data = "123.4567"
        try:
            atomic_write_text(f2, data)
            with open(f2, "r", encoding="utf-8") as r:
                content = r.read().strip()
            if content != data:
                print(f"[SELF-TEST] atomic_write_text wrote '{content}', expected '{data}'", file=sys.stderr)
                ok = False
        except Exception as e:
            print(f"[SELF-TEST] atomic_write_text raised error: {e}", file=sys.stderr)
            ok = False
        # Test remove
        try:
            remove_file(f1)
            if f1.exists():
                print("[SELF-TEST] remove_file did not remove file", file=sys.stderr)
                ok = False
        except Exception as e:
            print(f"[SELF-TEST] remove_file raised error: {e}", file=sys.stderr)
            ok = False
    if ok:
        print("[SELF-TEST] All file I/O tests PASSED")
        return 0
    else:
        print("[SELF-TEST] Some file I/O tests FAILED", file=sys.stderr)
        return 1


# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file. Returns an empty dict on failure."""
    if not yaml:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively. Overrides values in base with
    overrides. Used for CLI overrides of config values."""
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


# -----------------------------------------------------------------------------
# MIDI sidecar implementation
# -----------------------------------------------------------------------------

class MidiSidecar:
    """
    The MidiSidecar listens on a MIDI input port and writes macro files to
    disk according to configured note/CC mappings. It handles reconnection
    gracefully and supports various runtime overrides via CLI.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.running = True
        self.logger = logging.getLogger("midi_sidecar")
        # Apply log level from CLI or config
        log_level_str = args.log_level or config.get("midi", {}).get("log_level", "INFO")
        level = getattr(logging, str(log_level_str).upper(), logging.INFO)
        self.logger.setLevel(level)

        # Determine output directory
        cfg_midi = config.get("midi", {})
        default_dir = cfg_midi.get("live_output_dir", "live_output")
        out_dir_arg = args.out_dir or default_dir
        self.out_dir = Path(out_dir_arg).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Note off cleanup
        self.note_off_cleanup = args.note_off_cleanup if args.note_off_cleanup is not None else cfg_midi.get("note_off_cleanup", True)

        # Default note mode
        self.default_note_mode = args.default_note_mode or cfg_midi.get("default_note_mode")

        # Determine mapping
        mapping_cfg = cfg_midi.get("mapping", {})
        # Build note mapping: note number -> (action, mode)
        self.note_mapping: Dict[int, Dict[str, str]] = {}
        for note_str, entry in mapping_cfg.get("notes", {}).items():
            try:
                note_num = int(note_str)
            except Exception:
                continue
            if isinstance(entry, dict):
                action = str(entry.get("action", "")).strip().upper()
                mode = str(entry.get("mode", self.default_note_mode or "momentary")).strip().lower()
            else:
                # Backwards compatibility: if entry is a string, treat as action with momentary
                action = str(entry).strip().upper()
                mode = str(self.default_note_mode or "momentary").lower()
            if action:
                self.note_mapping[note_num] = {"action": action, "mode": mode}

        # Build CC mapping: cc number -> dict with action, min, max, smoothing
        self.cc_mapping: Dict[int, Dict[str, float | str]] = {}
        for cc_str, entry in mapping_cfg.get("cc", {}).items():
            try:
                cc_num = int(cc_str)
            except Exception:
                continue
            if isinstance(entry, dict):
                action = str(entry.get("action", "")).strip().upper()
                try:
                    min_v = float(entry.get("min", 0.0))
                    max_v = float(entry.get("max", 1.0))
                except Exception:
                    min_v, max_v = 0.0, 1.0
                try:
                    smoothing = float(entry.get("smoothing", 0.0))
                except Exception:
                    smoothing = 0.0
            else:
                # Backwards compatibility: if entry is a string, treat as action with default range 0–1 and no smoothing
                action = str(entry).strip().upper()
                min_v, max_v, smoothing = 0.0, 1.0, 0.0
            if action:
                self.cc_mapping[cc_num] = {
                    "action": action,
                    "min": min_v,
                    "max": max_v,
                    "smoothing": smoothing,
                    "value": None,  # For smoothing state
                }

        # Determine poll sleep (ms)
        self.poll_sleep = max(1, int(args.poll_sleep_ms or cfg_midi.get("poll_sleep_ms", 2))) / 1000.0

        # Determine port selection
        self.port_name: Optional[str] = None
        self.port_exact = args.port or cfg_midi.get("in_port")
        self.port_substring = args.name or cfg_midi.get("in_name")

        # Verbose CC logging
        self.verbose_cc = args.verbose_cc
        # Dry run flag
        self.dry_run = args.dry_run

    # ------------------------- Port Selection ---------------------------------

    def list_ports(self) -> None:
        """Print available MIDI input ports."""
        if not mido:
            self.logger.error("mido library not available. Install mido and python-rtmidi.")
            return
        ports = mido.get_input_names()
        if not ports:
            print("No MIDI input ports detected.")
        else:
            print("Available MIDI input ports:")
            for idx, p in enumerate(ports):
                print(f"  {idx}: {p}")

    def select_port(self) -> Optional[str]:
        """Select a MIDI input port based on CLI and config preferences."""
        if not mido:
            return None
        ports = mido.get_input_names()
        if not ports:
            self.logger.error("No MIDI input devices found.")
            return None
        # Exact port has highest priority
        if self.port_exact:
            for p in ports:
                if p == self.port_exact:
                    return p
            self.logger.error(f"MIDI port '{self.port_exact}' not found. Use --list to see available ports.")
            return None
        # Substring match
        if self.port_substring:
            for p in ports:
                if self.port_substring.lower() in p.lower():
                    return p
            self.logger.error(f"No MIDI input containing '{self.port_substring}' found. Use --list to see ports.")
            return None
        # If only one port, return it
        if len(ports) == 1:
            return ports[0]
        # Ambiguous: ask user to specify
        self.logger.error("Multiple MIDI input ports detected; please specify --port or --name. Use --list to view ports.")
        return None

    # --------------------------- Message Handling -----------------------------

    def _macro_path(self, action: str) -> Path:
        """Return the macro file path for a given action."""
        action_upper = action.upper()
        if action_upper == "AUDIO":
            return self.out_dir / "macro_AUDIO"
        elif action_upper == "DROP":
            return self.out_dir / "macro_DROP"
        elif action_upper == "BUILD":
            return self.out_dir / "macro_BUILD"
        elif action_upper == "CHILL":
            return self.out_dir / "macro_CHILL"
        elif action_upper == "INTENSITY":
            return self.out_dir / "macro_INTENSITY"
        elif action_upper == "ENERGY":
            return self.out_dir / "macro_ENERGY"
        else:
            # Unknown actions map to file named by action directly
            return self.out_dir / f"macro_{action_upper}"

    def handle_note_on(self, note: int, velocity: int) -> None:
        """Handle a note_on message with velocity > 0."""
        mapping = self.note_mapping.get(note)
        if not mapping:
            self.logger.debug(f"Ignoring note {note}: unmapped")
            return
        action = mapping.get("action")
        mode = mapping.get("mode", self.default_note_mode or "momentary").lower()
        path = self._macro_path(action)
        if mode == "toggle":
            exists = path.exists()
            if self.dry_run:
                self.logger.info(f"[DRY] NOTE {note} toggle {action}: would {'remove' if exists else 'create'} {path.name}")
            else:
                if exists:
                    remove_file(path)
                else:
                    touch_file(path)
                self.logger.info(f"NOTE {note} → toggle {action} -> {'OFF' if exists else 'ON'}")
        else:  # momentary
            if self.dry_run:
                self.logger.info(f"[DRY] NOTE {note} momentary {action}: would create/touch {path.name}")
            else:
                touch_file(path)
                self.logger.info(f"NOTE {note} → {action} ON")

    def handle_note_off(self, note: int) -> None:
        """Handle a note_off message or note_on with velocity 0."""
        mapping = self.note_mapping.get(note)
        if not mapping:
            return
        action = mapping.get("action")
        mode = mapping.get("mode", self.default_note_mode or "momentary").lower()
        if mode == "toggle":
            # Ignore note_off for toggle mappings
            return
        if not self.note_off_cleanup:
            return
        path = self._macro_path(action)
        if self.dry_run:
            self.logger.info(f"[DRY] NOTE OFF {note} momentary {action}: would remove {path.name}")
        else:
            remove_file(path)
            self.logger.info(f"NOTE {note} OFF → {action} OFF")

    def handle_cc(self, control: int, value: int) -> None:
        """Handle a control change message."""
        entry = self.cc_mapping.get(control)
        if not entry:
            self.logger.debug(f"Ignoring CC {control}: unmapped")
            return
        try:
            norm = max(0.0, min(1.0, value / 127.0))
            min_v = float(entry["min"])
            max_v = float(entry["max"])
            smoothing = float(entry["smoothing"])
            mapped = min_v + (max_v - min_v) * norm
            # Apply smoothing
            prev = entry.get("value")
            if prev is None or smoothing <= 0.0:
                smoothed = mapped
            else:
                smoothed = prev * (1.0 - smoothing) + mapped * smoothing
            # Store smoothed value
            entry["value"] = smoothed
            action = entry["action"]
            path = self._macro_path(action)
            if self.dry_run:
                self.logger.info(f"[DRY] CC {control} {action}: would write {smoothed:.4f} to {path.name}")
            else:
                atomic_write_text(path, f"{smoothed:.4f}")
                if self.verbose_cc:
                    self.logger.info(f"CC {control} → {action} {smoothed:.3f}")
                else:
                    self.logger.debug(f"CC {control} → {action} {smoothed:.3f}")
        except Exception as e:
            self.logger.warning(f"Failed to handle CC {control}: {e}")

    # -------------------------- Main Loop ------------------------------------

    def run(self) -> None:
        """Run the sidecar until interrupted. Handles device reconnection."""
        if not mido:
            self.logger.error("mido library not available. Install mido and python-rtmidi.")
            return
        # Determine the port once; if selection fails, exit
        self.port_name = self.select_port()
        if not self.port_name:
            return
        # Print mapping summary
        if self.note_mapping:
            for note, m in self.note_mapping.items():
                self.logger.info(f"Mapping NOTE {note} → {m['action']} ({m['mode']})")
        if self.cc_mapping:
            for cc, m in self.cc_mapping.items():
                self.logger.info(f"Mapping CC {cc} → {m['action']} range[{m['min']}, {m['max']}] smoothing {m['smoothing']}")
        self.logger.info(f"Macro directory: {self.out_dir}")
        attempt_delay = 0.5  # initial backoff
        while self.running:
            try:
                with mido.open_input(self.port_name) as inport:
                    self.logger.info(f"Connected to MIDI input: {self.port_name}")
                    attempt_delay = 0.5  # reset delay after successful connection
                    while self.running:
                        try:
                            pending = False
                            for msg in inport.iter_pending():
                                pending = True
                                # Determine message type
                                if msg.type == "note_on":
                                    # velocity 0 on note_on counts as note_off
                                    if msg.velocity == 0:
                                        self.handle_note_off(msg.note)
                                    else:
                                        self.handle_note_on(msg.note, msg.velocity)
                                elif msg.type == "note_off":
                                    self.handle_note_off(msg.note)
                                elif msg.type == "control_change":
                                    self.handle_cc(msg.control, msg.value)
                            if not pending:
                                time.sleep(self.poll_sleep)
                        except (OSError, IOError) as e:
                            # Device disconnected or error reading. Break to outer loop to reconnect.
                            self.logger.warning(f"MIDI input error: {e}. Reconnecting...")
                            break
            except Exception as e:
                self.logger.warning(f"Failed to open MIDI port '{self.port_name}': {e}")
            # Sleep before attempting to reconnect
            if not self.running:
                break
            self.logger.info(f"Retrying MIDI port {self.port_name} in {attempt_delay:.1f}s...")
            time.sleep(attempt_delay)
            attempt_delay = min(attempt_delay * 2.0, 10.0)

    # -------------------------- Shutdown -------------------------------------

    def stop(self, *_args) -> None:
        """Signal the sidecar to stop running."""
        self.running = False


# -----------------------------------------------------------------------------
# Argument parsing and entry point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual FaQtory MIDI sidecar")
    parser.add_argument(
        "--config",
        default="worqspace/config.yaml",
        help="Path to YAML config (default: worqspace/config.yaml)",
    )
    parser.add_argument("--list", action="store_true", help="List MIDI input ports and exit")
    parser.add_argument("--port", help="Exact MIDI input port to use")
    parser.add_argument("--name", help="Substring match for MIDI input port (case‑insensitive)")
    parser.add_argument("--out-dir", help="Directory where macro files are written")
    parser.add_argument("--log-level", help="Logging level: DEBUG, INFO, WARN, ERROR")
    parser.add_argument("--dry-run", action="store_true", help="Do not create/remove/write files; log actions only")
    parser.add_argument("--note-off-cleanup", type=lambda x: x.lower() == "true", nargs="?", const=True, default=None,
                        help="Remove momentary macro files on note off (true/false)")
    parser.add_argument("--default-note-mode", choices=["momentary", "toggle"], help="Default note mode when not specified in config")
    parser.add_argument("--poll-sleep-ms", type=int, help="Milliseconds to sleep when no messages (overrides config)")
    parser.add_argument("--verbose-cc", action="store_true", help="Log CC updates at INFO level instead of DEBUG")
    parser.add_argument("--self-test", action="store_true", help="Run file I/O self test and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Self test mode
    if args.self_test:
        sys.exit(run_self_test())
    # Load config if available
    config_path = Path(args.config).resolve()
    config: Dict[str, Any] = load_config(config_path) if config_path.exists() else {}
    # If --list, just list ports and exit
    if args.list:
        if not mido:
            print("mido library not available. Install mido and python-rtmidi.")
            return
        ports = mido.get_input_names()
        if not ports:
            print("No MIDI input ports detected.")
        else:
            print("Available MIDI input ports:")
            for idx, p in enumerate(ports):
                print(f"  {idx}: {p}")
        return
    # Merge CLI overrides into config? CLI overrides handled in constructor
    sidecar = MidiSidecar(config, args)
    # Register signal handlers
    signal.signal(signal.SIGINT, sidecar.stop)
    signal.signal(signal.SIGTERM, sidecar.stop)
    try:
        sidecar.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()