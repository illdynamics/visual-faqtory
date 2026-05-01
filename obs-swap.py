#!/usr/bin/env python3
"""
obs-swap.py — OBS A/B Aligned Swap (with legacy immediate prewarm)
═══════════════════════════════════════════════════════════════════════════
Performs a glitch-free A/B video source swap in OBS via WebSocket, aligned
to the natural end of the currently-visible clip so visuals don't jump
mid-loop.

DEFAULT BEHAVIOUR (aligned, v0.9.2+):
  1. Keep current slot Y on top, viewable.
  2. Place target slot X underneath Y, enable it, but park it parked at 0
     (stop + cursor=0 + pause if needed) so it doesn't visibly play
     underneath during the wait.
  3. Turn OFF looping on Y so it ends naturally.
  4. Wait for Y to report OBS_MEDIA_STATE_ENDED (or duration-cursor close
     to zero), up to OBS_MAX_WAIT_CURRENT_SEC.
  5. At that boundary: turn ON looping on X, force restart from frame 0,
     wait until it reports PLAYING (up to OBS_PREWARM_SEC).
  6. Disable Y, move X to top.
  7. Restore Y to a safe state (loop ON, stopped) so the next swap can
     reverse direction cleanly.

LEGACY IMMEDIATE BEHAVIOUR (--immediate or OBS_SWAP_MODE=immediate):
  Old prewarm-swap-on-arrival behaviour. Use only if you know you want it.

Environment / CLI overrides:
  OBS_HOST                  WebSocket host             (default: 127.0.0.1)
  OBS_PORT                  WebSocket port             (default: 4455)
  OBS_PASSWORD              WebSocket password         (default: Setyup34!)
  OBS_SCENE                 Scene name
  OBS_PREWARM_SEC           Target-PLAYING wait (s)    (default: 0.8)
  OBS_MAX_WAIT_CURRENT_SEC  Max wait for current end   (default: 180)
  OBS_END_THRESHOLD_MS      "Effectively ended" margin (default: 200)
  OBS_SWAP_MODE             "aligned" | "immediate"    (default: aligned)

Usage:
  python obs-swap.py A|B [--prewarm <s>] [--max-wait-current <s>]
                         [--end-threshold-ms <ms>] [--immediate|--aligned]

Assumption: the OBS media-source loop setting key is "looping" (canonical
name in obs-studio's ffmpeg_source plugin). If your OBS build uses a
different key, adjust LOOPING_KEY below.

Part of Visual FaQtory v0.9.2-beta
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[obs-swap] %(levelname)s: %(message)s",
)

# ── Configuration (overridable via environment) ───────────────────────────
HOST     = os.environ.get("OBS_HOST",     "127.0.0.1")
PORT     = int(os.environ.get("OBS_PORT", "4455"))
PASSWORD = os.environ.get("OBS_PASSWORD", "Setyup34!")
SCENE    = os.environ.get("OBS_SCENE",    "Ill Dynamics - Live on SkankOut")

# Source names — keep consistent with your OBS setup
SOURCE_A = "Live-Visuals-A"
SOURCE_B = "Live-Visuals-B"

ID_A = 7
ID_B = 8

DEFAULT_PREWARM_SEC          = float(os.environ.get("OBS_PREWARM_SEC",          "0.8"))
DEFAULT_MAX_WAIT_CURRENT_SEC = float(os.environ.get("OBS_MAX_WAIT_CURRENT_SEC", "180"))
DEFAULT_END_THRESHOLD_MS     = int(  os.environ.get("OBS_END_THRESHOLD_MS",     "200"))
DEFAULT_SWAP_MODE            =       os.environ.get("OBS_SWAP_MODE",            "aligned").strip().lower()

# Poll interval for media-state checks
POLL_INTERVAL = 0.05

# OBS media-input action constants
ACTION_RESTART = "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
ACTION_STOP    = "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_STOP"
ACTION_PAUSE   = "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PAUSE"
ACTION_PLAY    = "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PLAY"

# OBS source-settings key for looping (ffmpeg_source plugin canonical name).
# If your OBS build uses a different key (e.g. older builds with
# "loop" on vlc_source), override here.
LOOPING_KEY = "looping"


# ── Generic helpers ───────────────────────────────────────────────────────

def set_enabled(cl, scene: str, item_id: int, enabled: bool) -> None:
    try:
        cl.set_scene_item_enabled(scene, item_id, enabled)
    except Exception as e:
        logger.warning(f"set_scene_item_enabled({item_id}, {enabled}) failed: {e}")


def set_item_index(cl, scene: str, item_id: int, index: int) -> None:
    """Set scene item z-order index. 0 = top (rendered last / on top)."""
    try:
        cl.set_scene_item_index(scene, item_id, index)
    except Exception as e:
        logger.warning(f"set_scene_item_index({item_id}, {index}) failed: {e}")


# ── Media control helpers ─────────────────────────────────────────────────

def set_media_looping(cl, input_name: str, looping: bool) -> None:
    """Toggle the 'looping' setting on an OBS media input.

    Uses overlay=True so other input settings (local_file, restart_on_active,
    etc.) are NOT wiped. Best-effort: failures are logged but do not abort
    the swap — a live show is more important than a perfect log.
    """
    try:
        cl.set_input_settings(input_name, {LOOPING_KEY: bool(looping)}, overlay=True)
        logger.info(f"set_media_looping('{input_name}', {looping})")
    except Exception as e:
        logger.warning(f"set_media_looping('{input_name}', {looping}) failed: {e}")


def trigger_media_action(cl, input_name: str, action: str) -> bool:
    """Fire an OBS media-input action (RESTART / STOP / PAUSE / PLAY).

    Returns True on success, False on failure. Failures are logged but
    never raise — caller decides whether to continue.
    """
    try:
        cl.trigger_media_input_action(input_name, action)
        logger.debug(f"trigger_media_action('{input_name}', {action}) ok")
        return True
    except Exception as e:
        logger.warning(f"trigger_media_action('{input_name}', {action}) failed: {e}")
        return False


def get_media_status(cl, input_name: str):
    """Return (state, cursor_ms, duration_ms) for an OBS media input.

    Defensively reads both snake_case (current obsws_python) and camelCase
    (older / direct WebSocket) attribute names. Returns a 3-tuple of
    (str|None, int|None, int|None). On error, returns (None, None, None)
    and logs at debug level so polling loops aren't noisy.
    """
    try:
        resp = cl.get_media_input_status(input_name)
    except Exception as e:
        logger.debug(f"get_media_input_status('{input_name}') failed: {e}")
        return (None, None, None)

    state    = (getattr(resp, "media_state",    None) or
                getattr(resp, "mediaState",     None))
    cursor   = (getattr(resp, "media_cursor",   None) if hasattr(resp, "media_cursor") else
                getattr(resp, "mediaCursor",    None))
    duration = (getattr(resp, "media_duration", None) if hasattr(resp, "media_duration") else
                getattr(resp, "mediaDuration",  None))

    return (state, cursor, duration)


def wait_for_playing(cl, input_name: str, timeout_sec: float) -> bool:
    """Poll until the source reports PLAYING or cursor > 0. Fail-open."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        state, cursor, _ = get_media_status(cl, input_name)
        if state == "OBS_MEDIA_STATE_PLAYING" or (cursor is not None and cursor > 0):
            logger.info(
                f"'{input_name}' is playing (state={state}, cursor={cursor})"
            )
            return True
        time.sleep(POLL_INTERVAL)
    logger.warning(
        f"'{input_name}' did not reach PLAYING within {timeout_sec:.2f}s — "
        "proceeding anyway (fail-open)"
    )
    return False


def wait_for_media_ended(
    cl,
    input_name: str,
    max_wait_sec: float,
    end_threshold_ms: int,
) -> bool:
    """Poll until the source naturally ends, or the cursor is within
    `end_threshold_ms` of `duration` (which is the practical "frame before
    end" margin OBS reports).

    Logs status at the start and on success. On timeout, returns False so
    the caller can fail-open and proceed with the swap rather than
    deadlocking the live system.
    """
    state0, cursor0, duration0 = get_media_status(cl, input_name)
    logger.info(
        f"wait_for_media_ended('{input_name}'): start "
        f"state={state0} cursor={cursor0} duration={duration0} "
        f"max_wait={max_wait_sec:.1f}s end_threshold={end_threshold_ms}ms"
    )

    deadline = time.monotonic() + max_wait_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        state, cursor, duration = get_media_status(cl, input_name)

        # Primary success: OBS explicitly reports ENDED.
        if state == "OBS_MEDIA_STATE_ENDED":
            logger.info(
                f"'{input_name}' ENDED naturally (state={state}, "
                f"cursor={cursor}/{duration})"
            )
            return True

        # Backup success: cursor sits within end_threshold of duration.
        # Some media types (esp. when looping was just turned off mid-clip)
        # don't always emit STATE_ENDED cleanly.
        if (
            isinstance(cursor, (int, float)) and cursor > 0
            and isinstance(duration, (int, float)) and duration > 0
            and (duration - cursor) <= end_threshold_ms
        ):
            logger.info(
                f"'{input_name}' effectively ended "
                f"(cursor={cursor}/{duration}, within {end_threshold_ms}ms)"
            )
            return True

        # Periodic progress log (every ~5s) so a long-running wait isn't silent.
        now = time.monotonic()
        if now - last_log >= 5.0:
            logger.info(
                f"…still waiting on '{input_name}' "
                f"(state={state}, cursor={cursor}, duration={duration})"
            )
            last_log = now

        time.sleep(POLL_INTERVAL)

    logger.warning(
        f"wait_for_media_ended('{input_name}') TIMED OUT after "
        f"{max_wait_sec:.1f}s — failing OPEN to keep the show running"
    )
    return False


def park_target_at_start(cl, target_name: str) -> None:
    """Best-effort park: stop, seek to 0, optionally pause.

    Used after we enable the target source underneath the current visible
    one, to make sure the target doesn't visibly play behind the current
    clip while we're waiting for the boundary.

    Library-version-tolerant: each step is wrapped individually so a
    missing method on older obsws_python doesn't crash the show.
    """
    # 1. STOP — most reliable way to halt playback regardless of state.
    trigger_media_action(cl, target_name, ACTION_STOP)

    # 2. Seek cursor to 0. Method name in obsws_python is
    # set_media_input_cursor; if absent, swallow.
    try:
        if hasattr(cl, "set_media_input_cursor"):
            cl.set_media_input_cursor(target_name, 0)
            logger.info(f"park_target_at_start: cursor reset to 0 on '{target_name}'")
        else:
            logger.debug(
                "park_target_at_start: set_media_input_cursor not available "
                "in this obsws_python build (skipping seek)"
            )
    except Exception as e:
        logger.warning(f"park_target_at_start: cursor seek failed on '{target_name}': {e}")

    # 3. Pause is a no-op after STOP on most media types but harmless and
    # adds belt-and-braces against any auto-play behaviour the source
    # picks up from a settings change.
    trigger_media_action(cl, target_name, ACTION_PAUSE)


def force_media_reload(cl, input_name: str, do_stop_first: bool = False) -> None:
    """Force OBS to reload the media file via WebSocket.

    Args:
        do_stop_first: when True, STOP + cursor=0 + RESTART. Used at the
                       boundary-start to guarantee a clean from-frame-0 begin
                       (some OBS builds will otherwise pick up where a parked
                       target left off when looping flips on).

    Falls back to toggling local_file if RESTART fails (e.g. very old
    OBS WebSocket plugin).
    """
    if do_stop_first:
        trigger_media_action(cl, input_name, ACTION_STOP)
        try:
            if hasattr(cl, "set_media_input_cursor"):
                cl.set_media_input_cursor(input_name, 0)
        except Exception as e:
            logger.debug(f"pre-restart cursor reset failed on '{input_name}': {e}")

    if trigger_media_action(cl, input_name, ACTION_RESTART):
        logger.info(f"Media RESTART triggered for '{input_name}'")
        return

    # Fallback: blank then re-write local_file to force OBS to re-open.
    try:
        resp = cl.get_input_settings(input_name)
        settings = getattr(resp, "input_settings", {}) or {}
        local_file = settings.get("local_file", "")
        if local_file:
            cl.set_input_settings(input_name, {"local_file": ""}, overlay=True)
            time.sleep(0.05)
            cl.set_input_settings(input_name, {"local_file": local_file}, overlay=True)
            logger.info(f"Fallback: toggled local_file for '{input_name}'")
    except Exception as e2:
        logger.warning(f"Fallback media reload also failed for '{input_name}': {e2}")


# ── Swap implementations ──────────────────────────────────────────────────

def _resolve_slots(slot: str):
    """Map a slot letter to (target_name, target_id, current_name, current_id)."""
    if slot == "A":
        return SOURCE_A, ID_A, SOURCE_B, ID_B
    return SOURCE_B, ID_B, SOURCE_A, ID_A


def _connect():
    try:
        from obsws_python import ReqClient
        return ReqClient(host=HOST, port=PORT, password=PASSWORD)
    except Exception as e:
        logger.error(f"Failed to connect to OBS WebSocket: {e}")
        sys.exit(1)


def aligned_swap(slot: str, prewarm_sec: float, max_wait_current_sec: float,
                 end_threshold_ms: int) -> int:
    """Aligned A/B swap — wait for current to end, then start target from 0."""
    target_name, target_id, current_name, current_id = _resolve_slots(slot)

    logger.info(
        f"ALIGNED swap → {slot} "
        f"(target='{target_name}', current='{current_name}', "
        f"prewarm={prewarm_sec:.2f}s, "
        f"max_wait_current={max_wait_current_sec:.1f}s, "
        f"end_threshold={end_threshold_ms}ms)"
    )

    cl = _connect()

    # Step 1 — z-order: current on top, target hidden behind.
    logger.info("Step 1: z-order — current top (0), target below (1)")
    set_item_index(cl, SCENE, current_id, 0)
    set_item_index(cl, SCENE, target_id,  1)

    # Step 2 — both enabled. Current already visible; target underneath.
    # Enabling a media source can cause it to start playing immediately.
    # We park it at 0 below to keep it silent under the curtain.
    logger.info("Step 2: enabling target underneath current")
    set_enabled(cl, SCENE, current_id, True)
    set_enabled(cl, SCENE, target_id,  True)

    # Step 3 — Prepare target: looping OFF, force reload so OBS picks up the
    # new file, then PARK at 0 so it doesn't bleed underneath.
    logger.info("Step 3: prepare target (looping off, reload, park at 0)")
    set_media_looping(cl, target_name, False)
    force_media_reload(cl, target_name)
    # Tiny breath so OBS finishes opening the file before we park it.
    time.sleep(0.1)
    park_target_at_start(cl, target_name)

    # Step 4 — Turn OFF looping on current so it ends naturally.
    logger.info("Step 4: looping OFF on current — letting it finish naturally")
    set_media_looping(cl, current_name, False)

    # Step 5 — Block until current ends (or fail-open on timeout).
    logger.info(f"Step 5: waiting for '{current_name}' to end "
                f"(max {max_wait_current_sec:.1f}s)")
    ended_cleanly = wait_for_media_ended(
        cl, current_name, max_wait_current_sec, end_threshold_ms
    )
    if not ended_cleanly:
        logger.warning(
            "Current source did not signal END within budget — failing open "
            "to keep the live show running. Swap will proceed immediately."
        )

    # Step 6 — Boundary start: looping ON for target, hard restart from 0,
    # confirm PLAYING.
    logger.info("Step 6: boundary start — looping ON, restart, await PLAYING")
    set_media_looping(cl, target_name, True)
    force_media_reload(cl, target_name, do_stop_first=True)
    wait_for_playing(cl, target_name, prewarm_sec)

    # Step 7 — Reveal: disable current, target to top.
    logger.info("Step 7: revealing target — disabling current, target to z=0")
    set_enabled(cl, SCENE, current_id, False)
    set_item_index(cl, SCENE, target_id, 0)

    # Step 8 — Restore old current to a safe inactive state for next swap:
    # looping ON (so when it becomes target later it'll naturally loop until
    # we explicitly turn looping off again), and STOP so it doesn't keep
    # ticking under the new visible target.
    logger.info("Step 8: restoring old current to safe inactive state "
                "(loop ON, stopped)")
    set_media_looping(cl, current_name, True)
    trigger_media_action(cl, current_name, ACTION_STOP)

    logger.info(f"✓ Aligned swap complete — now active: {slot}")
    print(f"Switched to {slot}")
    return 0


def immediate_swap(slot: str, prewarm_sec: float) -> int:
    """LEGACY immediate prewarm swap — kept for backwards compatibility.

    Identical to the v0.9.1 behaviour: prewarm target, swap visibility as
    soon as it's playing. Use only when explicitly opted in via --immediate
    or OBS_SWAP_MODE=immediate.
    """
    target_name, target_id, current_name, current_id = _resolve_slots(slot)

    logger.info(
        f"IMMEDIATE (legacy) prewarm swap → {slot} "
        f"(target='{target_name}', current='{current_name}', "
        f"prewarm={prewarm_sec:.2f}s)"
    )

    cl = _connect()

    logger.info("Step 1: z-ordering — current on top, target below")
    set_item_index(cl, SCENE, current_id, 0)
    set_item_index(cl, SCENE, target_id,  1)

    logger.info("Step 2: enabling target source (behind current)")
    set_enabled(cl, SCENE, target_id, True)

    logger.info("Step 3: forcing media reload on target")
    force_media_reload(cl, target_name)

    logger.info(f"Step 4: polling for PLAYING state (timeout={prewarm_sec:.2f}s)")
    wait_for_playing(cl, target_name, prewarm_sec)

    logger.info("Step 5: disabling current source — swap complete")
    set_enabled(cl, SCENE, current_id, False)

    logger.info("Step 6: reordering target to index 0 for next cycle")
    set_item_index(cl, SCENE, target_id, 0)

    logger.info(f"✓ Immediate swap complete — now active: {slot}")
    print(f"Switched to {slot}")
    return 0


def prewarm_swap(slot: str, prewarm_sec: float, *, mode: str = "aligned",
                 max_wait_current_sec: float = DEFAULT_MAX_WAIT_CURRENT_SEC,
                 end_threshold_ms: int = DEFAULT_END_THRESHOLD_MS) -> int:
    """Front-door entry. Routes to aligned/immediate based on `mode`."""
    slot = slot.upper()
    if slot not in ("A", "B"):
        logger.error(f"Invalid slot '{slot}'. Must be A or B.")
        return 1

    mode = (mode or "aligned").strip().lower()
    if mode == "immediate":
        return immediate_swap(slot, prewarm_sec)
    return aligned_swap(slot, prewarm_sec, max_wait_current_sec, end_threshold_ms)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBS A/B swap — aligned to clip end (default) or "
                    "immediate prewarm (legacy)",
    )
    parser.add_argument(
        "slot",
        nargs="?",
        default=None,
        help="Target slot to activate: A or B",
    )
    parser.add_argument(
        "--prewarm",
        type=float,
        default=DEFAULT_PREWARM_SEC,
        metavar="SECONDS",
        help=f"Target-PLAYING wait (default: {DEFAULT_PREWARM_SEC}, "
             "env: OBS_PREWARM_SEC)",
    )
    parser.add_argument(
        "--max-wait-current",
        type=float,
        default=DEFAULT_MAX_WAIT_CURRENT_SEC,
        metavar="SECONDS",
        help=f"Max wait for current source to end (default: "
             f"{DEFAULT_MAX_WAIT_CURRENT_SEC}, env: OBS_MAX_WAIT_CURRENT_SEC)",
    )
    parser.add_argument(
        "--end-threshold-ms",
        type=int,
        default=DEFAULT_END_THRESHOLD_MS,
        metavar="MS",
        help=f"Effective-ended margin (default: {DEFAULT_END_THRESHOLD_MS}, "
             "env: OBS_END_THRESHOLD_MS)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--aligned",
        dest="mode",
        action="store_const",
        const="aligned",
        help="Aligned swap mode (default — wait for current to end)",
    )
    mode_group.add_argument(
        "--immediate",
        dest="mode",
        action="store_const",
        const="immediate",
        help="LEGACY immediate prewarm swap (mid-clip jump)",
    )
    parser.set_defaults(mode=DEFAULT_SWAP_MODE)

    args = parser.parse_args()

    if not args.slot:
        parser.print_help()
        sys.exit(1)

    rc = prewarm_swap(
        args.slot,
        args.prewarm,
        mode=args.mode,
        max_wait_current_sec=args.max_wait_current,
        end_threshold_ms=args.end_threshold_ms,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
