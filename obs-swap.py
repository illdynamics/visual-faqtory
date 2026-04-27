#!/usr/bin/env python3
"""
obs-swap.py — OBS A/B Prewarm Swap
════════════════════════════════════════════════════════════════════════════
Performs a glitch-free A/B video source swap in OBS via WebSocket.

Prewarm sequence (no black frame flash):
  1. Keep current slot Y visible and on top.
  2. Enable target slot X *underneath* Y (hidden from viewer).
  3. Force media reload on X via WebSocket (no "Close file when inactive" needed).
  4. Poll OBS until X is actually PLAYING (timeout configurable).
  5. Disable Y — swap complete, X is now visible.
  6. Reorder X to index 0 (top) for the next cycle.

Environment / CLI overrides:
  OBS_HOST          WebSocket host      (default: 127.0.0.1)
  OBS_PORT          WebSocket port      (default: 4455)
  OBS_PASSWORD      WebSocket password  (default: Setyup34!)
  OBS_SCENE         Scene name
  OBS_PREWARM_SEC   Prewarm timeout in seconds (default: 0.8)

Usage:
  python obs-swap.py A|B [--prewarm <seconds>]

Part of Visual FaQtory v0.5.8-beta
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
SCENE    = os.environ.get("OBS_SCENE",    "Ill Dynamics - Live on Cyndicut Radio")

# Source names — keep consistent with your OBS setup
SOURCE_A = "Live-Visuals-A"
SOURCE_B = "Live-Visuals-B"

ID_A = 8
ID_B = 9

DEFAULT_PREWARM_SEC = float(os.environ.get("OBS_PREWARM_SEC", "0.8"))

# Poll interval for media-state check
POLL_INTERVAL = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────

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


def force_media_reload(cl, input_name: str) -> None:
    """Force OBS to reload the media file via WebSocket restart action.

    Removes the need for 'Close file when inactive' — we push a RESTART
    command so OBS re-opens the file from disk on each call.
    """
    try:
        cl.trigger_media_input_action(
            input_name,
            "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
        )
        logger.info(f"Media reload triggered for '{input_name}'")
    except Exception as e:
        logger.warning(f"trigger_media_input_action({input_name}) failed: {e}")
        # Fallback: toggle local_file setting to force re-open
        try:
            resp = cl.get_input_settings(input_name)
            settings = resp.input_settings
            local_file = settings.get("local_file", "")
            if local_file:
                cl.set_input_settings(input_name, {"local_file": ""}, overlay=True)
                time.sleep(0.05)
                cl.set_input_settings(input_name, {"local_file": local_file}, overlay=True)
                logger.info(f"Fallback: toggled local_file for '{input_name}'")
        except Exception as e2:
            logger.warning(f"Fallback media reload also failed for '{input_name}': {e2}")


def wait_for_playing(cl, input_name: str, timeout_sec: float) -> bool:
    """Poll OBS until the source reports PLAYING state or cursor > 0.

    Fails-open: returns False on timeout so the swap still completes.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            resp = cl.get_media_input_status(input_name)
            state = getattr(resp, "media_state", None) or ""
            cursor = getattr(resp, "media_cursor", None) or 0
            if state == "OBS_MEDIA_STATE_PLAYING" or (cursor is not None and cursor > 0):
                logger.info(
                    f"'{input_name}' is playing "
                    f"(state={state}, cursor={cursor})"
                )
                return True
        except Exception as e:
            logger.debug(f"get_media_input_status({input_name}) poll error: {e}")
        time.sleep(POLL_INTERVAL)
    logger.warning(
        f"'{input_name}' did not reach PLAYING state within {timeout_sec:.2f}s — "
        "proceeding anyway (fail-open)"
    )
    return False


# ── Main prewarm swap ─────────────────────────────────────────────────────

def prewarm_swap(slot: str, prewarm_sec: float) -> None:
    """Execute the prewarm A/B swap with no black frame flash.

    Args:
        slot:        Target slot to activate ("A" or "B").
        prewarm_sec: Maximum time to wait for the target source to start playing.
    """
    slot = slot.upper()
    if slot not in ("A", "B"):
        logger.error(f"Invalid slot '{slot}'. Must be A or B.")
        sys.exit(1)

    # Determine target (X) and current (Y) slots
    if slot == "A":
        target_name  = SOURCE_A
        target_id    = ID_A
        current_name = SOURCE_B
        current_id   = ID_B
    else:
        target_name  = SOURCE_B
        target_id    = ID_B
        current_name = SOURCE_A
        current_id   = ID_A

    logger.info(
        f"Prewarm swap → {slot} "
        f"(target='{target_name}', current='{current_name}', "
        f"prewarm={prewarm_sec:.2f}s)"
    )

    try:
        from obsws_python import ReqClient
        cl = ReqClient(host=HOST, port=PORT, password=PASSWORD)
    except Exception as e:
        logger.error(f"Failed to connect to OBS WebSocket: {e}")
        sys.exit(1)

    # Step 1: Put current (Y) on top, target (X) below during prewarm.
    #         Viewers only see Y while X is warming up underneath.
    logger.info("Step 1: z-ordering — current on top (0), target below (1)")
    set_item_index(cl, SCENE, current_id, 0)
    set_item_index(cl, SCENE, target_id,  1)

    # Step 2: Enable target X while it's hidden behind Y
    logger.info("Step 2: enabling target source (behind current)")
    set_enabled(cl, SCENE, target_id, True)

    # Step 3: Force media reload so OBS reads the new file from disk
    logger.info("Step 3: forcing media reload on target")
    force_media_reload(cl, target_name)

    # Step 4: Wait until target is actually playing
    logger.info(f"Step 4: polling for PLAYING state (timeout={prewarm_sec:.2f}s)")
    wait_for_playing(cl, target_name, prewarm_sec)

    # Step 5: Disable current Y — the swap becomes visible
    logger.info("Step 5: disabling current source — swap complete")
    set_enabled(cl, SCENE, current_id, False)

    # Step 6: Move target to top (index 0) so next prewarm works in reverse
    logger.info("Step 6: reordering target to index 0 for next cycle")
    set_item_index(cl, SCENE, target_id, 0)

    logger.info(f"✓ Swap complete — now active: {slot}")
    print(f"Switched to {slot}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBS A/B prewarm swap — no black frame flash",
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
        help=f"Prewarm timeout in seconds (default: {DEFAULT_PREWARM_SEC}, "
             "env: OBS_PREWARM_SEC)",
    )
    args = parser.parse_args()

    if not args.slot:
        parser.print_help()
        sys.exit(1)

    prewarm_swap(args.slot, args.prewarm)


if __name__ == "__main__":
    main()
