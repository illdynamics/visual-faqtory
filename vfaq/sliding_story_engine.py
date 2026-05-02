#!/usr/bin/env python3
"""
sliding_story_engine.py — Paragraph-Driven Sliding Window Story Engine
══════════════════════════════════════════════════════════════════════════════

This module implements a deterministic, paragraph-driven narrative engine
for the Visual FaQtory. It produces a sequence of keyframes and
transition videos from a plain text story. Paragraphs are
interpreted as discrete story steps and are combined into a rolling
window according to a ramp‑up → slide → ramp‑down schedule. Visual
continuity is achieved by seeding each generation with the last frame of
the previous cycle.

Key characteristics:

  • Story parsing, window logic and cycle control are entirely
    implemented on the Visual FaQtory side. The backend (ComfyUI or Veo)
    is never allowed to own directory structure or story state.

  • Backend operations:
    - ComfyUI: text2img → img2img → img2vid (traditional pipeline)
    - Veo: text_to_video / image_to_video / first_last_frame (direct)
    - LTX-Video: t2v / i2v / conditioned_transition (self-hosted, direct)

  • The runtime directories are flat: qodeyard/story.txt, qodeyard/keyframes,
    qodeyard/lastframes, qodeyard/videos. No subdirectories per run.

  • Veo-aware routing (v0.6.0-beta):
    - Cycle 1: text_to_video (or image_to_video if base image exists)
    - Cycle n>1: image_to_video from previous last frame
    - If require_morph: first_last_frame (start + end keyframe)
    - Optional loop closure: final cycle generates clip from last frame
      back to cycle-1 anchor frame
    - Cycle 1: t2v (or i2v if base image exists)
    - Cycle n>1: i2v from previous last frame (reinject semantics)
    - If require_morph: conditioned transition (two-keyframe conditioning)
    - Optional loop closure: same as Veo via generate_morph_video

  • The windowing schedule has three phases for P total paragraphs and
    maximum window size M (max_paragraphs):

        – Ramp‑up: cycles 1..M grow the window [1], [1,2], [1,2,3], …
        – Sliding: cycles M+1..P slide a fixed size window of M paragraphs
        – Ramp‑down: after the last paragraph triggers, continue cycles
          dropping the earliest paragraph until only one remains.

    Generation stops after the single‑paragraph window completes.

  • Denoise values for img2img are drawn uniformly from a configurable
    range [img2img_denoise_min, img2img_denoise_max] per cycle.
    (ComfyUI backend only; Veo does not use denoise values.)

To run the engine, construct a SlidingStoryConfig (see below) and call
run_sliding_story(). For convenience, a CLI entry point is provided
via vfaq_cli (see vfaq_cli.py).

Part of Visual FaQtory v0.7.0-beta
"""
from __future__ import annotations

import logging
import os
import random
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from vfaq.timing import TimingResolver
from vfaq.finalizer import Finalizer # Import Finalizer

import subprocess
import tempfile

from .backends import (
    create_backend,
    describe_backend_config,
    get_backend_type_for_capability,
    resolve_capability_backend_configs,
    GenerationRequest,
    GenerationResult,
    InputMode,
)
from .image_metrics import calculate_frame_similarity

logger = logging.getLogger(__name__)
_ACTIVE_SMART_REINJECT_STATES: List["_SmartReinjectState"] = []
_ACTIVE_SMART_REINJECT_STATES_LOCK = threading.Lock()


def _write_briq_json(briqs_dir: Path, cycle_idx: int, data: dict) -> None:
    """Write per-cycle briq JSON for reproducibility."""
    path = briqs_dir / f"cycle_{cycle_idx:03d}.json"
    import json
    path.write_text(json.dumps(data, indent=2, default=str))


@dataclass
class SlidingStoryConfig:
    """Configuration for the sliding window story engine.

    Attributes:
        max_paragraphs:      Maximum number of paragraphs to include in the
                             window. Controls ramp‑up/sliding length.
        img2vid_duration_sec: Optional duration in seconds of each transition video.
        img2img_denoise_min: Minimum denoise_strength for img2img calls.
        img2img_denoise_max: Maximum denoise_strength for img2img calls.
        rolling_window_mode: Enable the sliding window engine when True.
        require_morph:       When true, cycles ≥ 2 must use morphing.
        seed_base:           Base seed used for deterministic generation.
        video_fps:           Optional frames per second for cross‑fade videos.
        video_frames:        Optional number of frames for cross-fade videos.
        timing_authority:    Optional timing authority ("frames", "duration", or "fps").
        backend_config:      Backend configuration dictionary used to
                             instantiate the generator backend.
        veo_config:          Optional dict with Veo-specific settings.
        ltx_video_config:    Optional dict with LTX-Video-specific settings (v0.6.7-beta).
        venice_config:       Optional dict with Venice-specific settings (v0.7.1-beta).
        enable_loop_closure: When true, generate a final loop-closure clip
                             that transitions from the last frame back to the
                             cycle-1 anchor frame.
        smart_reinject_*:    Optional async reinject controls (Venice-first).
                             Disabled by default to preserve v0.9.2 behavior.
    """
    max_paragraphs: int = 4
    img2vid_duration_sec: Optional[float] = None
    img2img_denoise_min: float = 0.25
    img2img_denoise_max: float = 0.45
    rolling_window_mode: bool = True
    require_morph: bool = False
    seed_base: int = 42
    video_fps: Optional[float] = None
    video_frames: Optional[int] = None
    timing_authority: Optional[str] = None
    backend_config: Dict[str, any] | None = None
    finalizer_config: Dict[str, any] | None = None # Add finalizer_config
    reinject: bool = True  # When True (default), every cycle runs img2img keyframe
    crowd_control_config: Dict[str, any] | None = None  # Crowd Control settings
    veo_config: Dict[str, any] | None = None  # Veo-specific config (v0.6.0-beta)
    venice_config: Dict[str, any] | None = None  # Venice config (v0.7.1-beta)
    enable_loop_closure: bool = False  # Generate final loop-closure clip (v0.6.0-beta)
    smart_reinject_enabled: bool = False
    smart_reinject_every_n_cycles: int = 1
    smart_reinject_use_morph: bool = True
    smart_reinject_similarity_guard_enabled: bool = True
    smart_reinject_similarity_threshold: float = 0.42
    smart_reinject_wait_timeout_sec: float = 0.0
    smart_reinject_sync_fallback: bool = False
    smart_reinject_denoise_min: Optional[float] = None
    smart_reinject_denoise_max: Optional[float] = None
    smart_reinject_prompt_prefix: str = (
        "Preserve the source image strongly. Make a subtle evolved keyframe for the next visual beat. "
        "Keep composition, identity, palette, lighting, and major shapes stable. Avoid large scene changes."
    )
    continuity_guard_enabled: bool = True  # Retry overly drifted conditioned videos
    continuity_similarity_threshold: float = 0.42  # 0..1, higher is stricter for generic conditioned video
    continuity_morph_similarity_threshold: float = 0.86  # Morph endpoints should land much closer to the target keyframe
    continuity_retry_attempts: int = 2  # Extra retries after the first render
    continuity_ffmpeg_fallback_enabled: bool = True  # Replace unrecoverable soup with a sane fallback clip
    continuity_ffmpeg_fallback_min_similarity: float = 0.32  # Legacy soft-accept floor for non-morph paths
    continuity_force_fallback_for_morph: bool = True  # Never propagate a weak morph endpoint into the next cycle


@dataclass
class _SmartReinjectPrefetch:
    future: Future
    source_cycle_idx: int
    target_cycle_idx: int
    source_image_path: Path
    atom_id: str
    denoise_strength: float
    prompt_preview: str


@dataclass
class _SmartReinjectState:
    enabled: bool = False
    executor: Optional[ThreadPoolExecutor] = None
    image_backend: Optional[Any] = None
    pending: Optional[_SmartReinjectPrefetch] = None
    abandoned_prefetches: List[_SmartReinjectPrefetch] = field(default_factory=list)
    warned_end_frame_morph_disabled: bool = False
    warned_morph_capability_failure: bool = False


@dataclass
class ResolvedCyclePrompt:
    base_story_prompt: str
    crowd_prompt: Optional[str]
    crowd_prompt_origin: str
    inject_mode: str
    inject_source_mode: str
    resolved_main_prompt: str
    crowd_active: bool


def _normalize_smart_reinject_interval(raw_value: Any) -> int:
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


def _normalize_crowd_carryover_cycles(raw_value: Any) -> int:
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _resolve_cycle_prompt(
    *,
    base_story_prompt: str,
    crowd_prompt: Optional[str],
    crowd_prompt_origin: str,
    inject_mode: str,
    inject_source_mode: str,
    inject_label: str,
) -> ResolvedCyclePrompt:
    normalized_mode = str(inject_mode or "append").strip().lower() or "append"
    if normalized_mode not in {"append", "replace"}:
        normalized_mode = "append"
    normalized_source_mode = str(inject_source_mode or "as_image_source").strip().lower() or "as_image_source"
    if normalized_source_mode not in {"as_image_source", "as_reference"}:
        normalized_source_mode = "as_image_source"

    crowd_text = str(crowd_prompt or "").strip()
    if not crowd_text:
        return ResolvedCyclePrompt(
            base_story_prompt=base_story_prompt,
            crowd_prompt=None,
            crowd_prompt_origin="none",
            inject_mode=normalized_mode,
            inject_source_mode=normalized_source_mode,
            resolved_main_prompt=base_story_prompt,
            crowd_active=False,
        )

    if normalized_mode == "replace":
        resolved_main_prompt = crowd_text
    else:
        label = str(inject_label or "Audience mutation request").strip().upper() or "AUDIENCE MUTATION REQUEST"
        resolved_main_prompt = base_story_prompt + "\n\n[" + label + "]\n" + crowd_text

    return ResolvedCyclePrompt(
        base_story_prompt=base_story_prompt,
        crowd_prompt=crowd_text,
        crowd_prompt_origin=crowd_prompt_origin if crowd_prompt_origin in {"fresh", "carryover"} else "fresh",
        inject_mode=normalized_mode,
        inject_source_mode=normalized_source_mode,
        resolved_main_prompt=resolved_main_prompt,
        crowd_active=True,
    )


def _safe_unlink(path: Optional[Path]) -> None:
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


def _abandon_smart_reinject_prefetch(
    *,
    state: _SmartReinjectState,
    prefetch: _SmartReinjectPrefetch,
    reason: str,
    cycle_idx: Optional[int] = None,
) -> None:
    if prefetch.future.done():
        try:
            result = prefetch.future.result()
            if result and getattr(result, "image_path", None):
                _safe_unlink(Path(result.image_path))
        except Exception as e:
            logger.warning(
                f"[SlidingStory/SmartReinject] Prefetch {prefetch.atom_id} "
                f"finished with error while abandoning ({reason}): {e}"
            )
        return

    cancelled = False
    try:
        cancelled = bool(prefetch.future.cancel())
    except Exception as e:
        logger.warning(
            f"[SlidingStory/SmartReinject] Could not cancel pending prefetch "
            f"{prefetch.atom_id} ({reason}): {e}"
        )

    if cancelled:
        logger.info(
            "[SlidingStory/SmartReinject] Cancelled pending prefetch "
            f"{prefetch.atom_id} ({reason})."
        )
        return

    state.abandoned_prefetches.append(prefetch)
    suffix = f" for cycle {cycle_idx}" if cycle_idx is not None else ""
    logger.info(
        "[SlidingStory/SmartReinject] Prefetch "
        f"{prefetch.atom_id}{suffix} marked stale/abandoned ({reason}) and will be ignored on completion."
    )


def _reap_abandoned_smart_reinject_prefetches(state: _SmartReinjectState) -> None:
    if not state.abandoned_prefetches:
        return
    remaining: List[_SmartReinjectPrefetch] = []
    for prefetch in state.abandoned_prefetches:
        if not prefetch.future.done():
            remaining.append(prefetch)
            continue
        try:
            result = prefetch.future.result()
            if result and getattr(result, "image_path", None):
                _safe_unlink(Path(result.image_path))
                logger.info(
                    "[SlidingStory/SmartReinject] Discarded late abandoned prefetch "
                    f"{prefetch.atom_id} after completion."
                )
        except Exception as e:
            logger.warning(
                f"[SlidingStory/SmartReinject] Abandoned prefetch {prefetch.atom_id} failed: {e}"
            )
    state.abandoned_prefetches = remaining


def _register_smart_reinject_state(state: _SmartReinjectState) -> None:
    with _ACTIVE_SMART_REINJECT_STATES_LOCK:
        if state not in _ACTIVE_SMART_REINJECT_STATES:
            _ACTIVE_SMART_REINJECT_STATES.append(state)


def _unregister_smart_reinject_state(state: _SmartReinjectState) -> None:
    with _ACTIVE_SMART_REINJECT_STATES_LOCK:
        if state in _ACTIVE_SMART_REINJECT_STATES:
            _ACTIVE_SMART_REINJECT_STATES.remove(state)


def shutdown_active_smart_reinject_workers(reason: str = "interrupt") -> None:
    with _ACTIVE_SMART_REINJECT_STATES_LOCK:
        states = list(_ACTIVE_SMART_REINJECT_STATES)
    for state in states:
        _shutdown_smart_reinject_state(state, reason=reason)


def _should_schedule_smart_reinject(
    *,
    state: _SmartReinjectState,
    config: SlidingStoryConfig,
    cycle_idx: int,
    total_cycles: int,
    effective_reinject: bool,
    crowd_active: bool,
    source_image_path: Optional[Path],
) -> bool:
    _reap_abandoned_smart_reinject_prefetches(state)
    if not state.enabled:
        return False
    if not effective_reinject:
        return False
    if crowd_active:
        return False
    if cycle_idx >= total_cycles:
        return False
    if source_image_path is None or not source_image_path.exists():
        return False
    if state.pending and not state.pending.future.done():
        logger.info(
            "[SlidingStory/SmartReinject] Previous prefetch still running "
            f"(target cycle {state.pending.target_cycle_idx}) — skip new schedule."
        )
        return False
    every_n = _normalize_smart_reinject_interval(config.smart_reinject_every_n_cycles)
    return ((cycle_idx - 1) % every_n) == 0


def _start_smart_reinject_prefetch(
    *,
    state: _SmartReinjectState,
    cycle_idx: int,
    target_cycle_idx: int,
    source_image_path: Path,
    prompt: str,
    seed: int,
    denoise_strength: float,
    width: int,
    height: int,
    output_dir: Path,
) -> bool:
    if not state.enabled or state.executor is None or state.image_backend is None:
        return False
    atom_id = f"smart_reinject_{cycle_idx:03d}_for_{target_cycle_idx:03d}"
    req = GenerationRequest(
        prompt=prompt,
        negative_prompt="",
        seed=seed,
        mode=InputMode.IMAGE,
        init_image_path=source_image_path,
        denoise_strength=denoise_strength,
        width=width,
        height=height,
        output_dir=output_dir,
        atom_id=atom_id,
    )
    future = state.executor.submit(state.image_backend.generate_image, req)
    state.pending = _SmartReinjectPrefetch(
        future=future,
        source_cycle_idx=cycle_idx,
        target_cycle_idx=target_cycle_idx,
        source_image_path=source_image_path,
        atom_id=atom_id,
        denoise_strength=denoise_strength,
        prompt_preview=prompt[:220],
    )
    logger.info(
        "[SlidingStory/SmartReinject] Scheduled async prefetch "
        f"{atom_id} ({source_image_path.name} -> cycle {target_cycle_idx})"
    )
    return True


def _collect_smart_reinject_prefetch(
    *,
    state: _SmartReinjectState,
    cycle_idx: int,
    keyframes_dir: Path,
    wait_timeout_sec: float,
) -> Optional[Dict[str, Any]]:
    _reap_abandoned_smart_reinject_prefetches(state)
    pending = state.pending
    if not pending:
        return None

    if pending.target_cycle_idx != cycle_idx:
        if pending.future.done():
            try:
                result = pending.future.result()
                if result and result.image_path and result.image_path.exists():
                    _safe_unlink(Path(result.image_path))
                    logger.info(
                        "[SlidingStory/SmartReinject] Discarding stale prefetch "
                        f"{pending.atom_id} (ready for cycle {pending.target_cycle_idx}, now cycle {cycle_idx})"
                    )
            except Exception as e:
                logger.warning(f"[SlidingStory/SmartReinject] Stale prefetch failed: {e}")
            finally:
                state.pending = None
        return None

    timeout = max(0.0, float(wait_timeout_sec or 0.0))
    if not pending.future.done():
        if timeout <= 0.0:
            logger.info(
                "[SlidingStory/SmartReinject] Prefetch not ready for cycle "
                f"{cycle_idx} and wait_timeout=0 — continuing without waiting."
            )
            _abandon_smart_reinject_prefetch(
                state=state,
                prefetch=pending,
                reason="missed_target_wait_timeout_zero",
                cycle_idx=cycle_idx,
            )
            state.pending = None
            logger.info(
                "[SlidingStory/SmartReinject] Smart prefetch missed target cycle; "
                "marked stale and will be ignored on completion."
            )
            return None
        try:
            pending.future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.info(
                "[SlidingStory/SmartReinject] Prefetch still not ready after "
                f"{timeout:.2f}s for cycle {cycle_idx} — continuing without waiting."
            )
            return None
        except Exception as e:
            logger.warning(f"[SlidingStory/SmartReinject] Prefetch failed for cycle {cycle_idx}: {e}")
            state.pending = None
            return None

    try:
        result = pending.future.result()
    except Exception as e:
        logger.warning(f"[SlidingStory/SmartReinject] Prefetch result failed for cycle {cycle_idx}: {e}")
        state.pending = None
        return None

    state.pending = None
    if not (result and result.success and result.image_path and result.image_path.exists()):
        logger.warning(
            f"[SlidingStory/SmartReinject] Prefetch returned no usable keyframe for cycle {cycle_idx}"
        )
        return None

    target_path = keyframes_dir / f"smart_reinject_target_{cycle_idx:03d}.png"
    try:
        if result.image_path != target_path:
            shutil.move(str(result.image_path), str(target_path))
        else:
            target_path = result.image_path
    except Exception as e:
        logger.warning(f"[SlidingStory/SmartReinject] Failed to materialize prefetch keyframe: {e}")
        return None

    logger.info(
        "[SlidingStory/SmartReinject] Prefetch ready for cycle "
        f"{cycle_idx} from cycle {pending.source_cycle_idx}: {target_path.name}"
    )
    return {
        "keyframe_path": target_path,
        "source_cycle_idx": pending.source_cycle_idx,
        "atom_id": pending.atom_id,
        "denoise_strength": pending.denoise_strength,
        "prompt_preview": pending.prompt_preview,
    }


def _discard_pending_smart_reinject_for_cycle(
    *,
    state: _SmartReinjectState,
    cycle_idx: int,
    reason: str,
) -> bool:
    pending = state.pending
    if pending is None or pending.target_cycle_idx != cycle_idx:
        return False

    _abandon_smart_reinject_prefetch(
        state=state,
        prefetch=pending,
        reason=reason,
        cycle_idx=cycle_idx,
    )
    state.pending = None
    logger.info(
        "[SlidingStory/SmartReinject] Discarded pending prefetch "
        f"{pending.atom_id} for cycle {cycle_idx} ({reason})."
    )
    return True


def _shutdown_smart_reinject_state(state: _SmartReinjectState, reason: str = "normal") -> None:
    _reap_abandoned_smart_reinject_prefetches(state)
    if not state.executor and not state.pending and not state.abandoned_prefetches:
        _unregister_smart_reinject_state(state)
        return
    if reason == "interrupt":
        logger.info(
            "[SlidingStory/SmartReinject] Interrupt received: cancelling/abandoning "
            "smart reinject prefetch workers."
        )
    try:
        if state.pending is not None:
            _abandon_smart_reinject_prefetch(
                state=state,
                prefetch=state.pending,
                reason="shutdown_interrupt" if reason == "interrupt" else "shutdown",
            )
            state.pending = None
        for prefetch in list(state.abandoned_prefetches):
            if not prefetch.future.done():
                try:
                    prefetch.future.cancel()
                except Exception:
                    pass
        if state.executor:
            state.executor.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        logger.warning(f"[SlidingStory/SmartReinject] Executor shutdown warning: {e}")
    finally:
        state.pending = None
        state.abandoned_prefetches = []
        state.executor = None
        _unregister_smart_reinject_state(state)
        logger.info("[SlidingStory/SmartReinject] Smart reinject executor shutdown complete.")

def _parse_story_file(story_path: Path) -> List[str]:
    """Parse a plain text story file into paragraphs.

    A paragraph is defined as a block of non‑empty lines separated by one
    or more blank lines. Leading/trailing whitespace is stripped from each
    paragraph. Empty paragraphs are ignored.

    Args:
        story_path: Path to the story.txt file.

    Returns:
        List of paragraph strings in order of appearance.
    """
    if not story_path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    content = story_path.read_text(encoding='utf-8')
    # Split on blank lines (two or more newline characters). Use simple
    # splitting because front‑matter is not used here.
    raw_paragraphs = [p.strip() for p in content.split("\n\n")]
    paragraphs: List[str] = []
    for para in raw_paragraphs:
        clean = para.strip()
        if clean:
            paragraphs.append(clean)
    return paragraphs


def _load_optional_prompt_text(base_dir: Path, filename: str) -> str:
    """Load an optional prompt-sidecar file from the story directory."""
    try:
        path = base_dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding='utf-8').strip()
    except Exception as e:
        logger.warning(f"[SlidingStory] Failed to read optional prompt file {filename}: {e}")
        return ""


def _build_video_stage_prompt(image_prompt: str, motion_prompt: str) -> str:
    """Build a dedicated video-stage prompt while keeping image prompt untouched."""
    if motion_prompt and motion_prompt.strip():
        return f"{image_prompt}\nMOTION: {motion_prompt.strip()}"
    return image_prompt


def _determine_windows(num_paragraphs: int, max_paragraphs: int) -> List[List[int]]:
    """Compute the sliding window schedule for the story.

    Given P paragraphs and maximum window size M, produce a list of lists
    where each inner list contains 1‑based indices of paragraphs to include
    for the corresponding cycle. The schedule follows three phases:

        1. Ramp‑up: For cycle 1..min(P, M), include paragraphs [1],
           [1,2], … up to [1..M].
        2. Sliding: For cycles M+1..P, include windows of size M that
           slide forward by one: [2..M+1], [3..M+2], … [P-M+1..P].
        3. Ramp‑down: After the last full window, drop the earliest
           paragraph each cycle until only one remains: [P-(M-2)..P], …,
           [P]. The final cycle is the single paragraph window.

    Args:
        num_paragraphs: Total number of paragraphs in the story.
        max_paragraphs: Maximum window size M.

    Returns:
        List of windows, each represented as a list of 1‑based indices.
    """
    windows: List[List[int]] = []
    M = max(1, max_paragraphs)
    P = num_paragraphs

    # Phase 1: ramp‑up — grow window from [1] to [1..min(P, M)]
    for k in range(1, min(P, M) + 1):
        windows.append(list(range(1, k + 1)))

    # Phase 2: sliding — when P > M, slide a fixed window of size M
    if P > M:
        for start in range(2, P - M + 2):
            windows.append(list(range(start, start + M)))

    # Phase 3: ramp‑down — always ramp down to a single paragraph
    # Start from the last generated window and drop the earliest paragraph
    if windows:
        last_window = windows[-1]
        # Continue dropping first element until only one remains
        while len(last_window) > 1:
            last_window = last_window[1:]
            windows.append(last_window.copy())

    return windows



def _resize_frame_to_target(path: Path, target_w: int, target_h: int) -> None:
    """
    Resize an extracted frame PNG in-place to (target_w, target_h) using Lanczos.
    Called after ffmpeg frame extraction when the video backend outputs a different
    resolution than the configured target (e.g. Seedance ignores input image dims).
    No-op if the frame is already the right size or if PIL is unavailable.
    """
    try:
        from PIL import Image
        img = Image.open(path)
        if img.size == (target_w, target_h):
            return
        logger.debug(
            f"[SlidingStory] Resizing extracted frame {img.size[0]}×{img.size[1]}"
            f" → {target_w}×{target_h}: {path.name}"
        )
        img.resize((target_w, target_h), Image.LANCZOS).save(str(path), "PNG")
    except Exception as e:
        logger.warning(f"[SlidingStory] Frame resize failed for {path.name}: {e}")


def _probe_video_metadata(video_path: Path) -> Dict[str, float | int | None]:
    """Probe basic video metadata with sane fallbacks.

    We intentionally ask ffprobe for both stream and format duration plus counted
    frames. Some generated files omit one of those fields; in that case we derive
    duration from frames/fps so downstream frame extraction does not go blind.
    """
    meta: Dict[str, float | int | None] = {
        "duration": 0.0,
        "fps": 0.0,
        "nb_frames": None,
    }
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate,duration',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        payload = json.loads(result.stdout or '{}')
        stream = (payload.get('streams') or [{}])[0]
        fmt = payload.get('format') or {}

        def _to_float(value):
            if value in (None, '', 'N/A'):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        fps_raw = stream.get('avg_frame_rate') or stream.get('r_frame_rate')
        if isinstance(fps_raw, str) and '/' in fps_raw:
            num, den = fps_raw.split('/', 1)
            try:
                den_f = float(den)
                if den_f:
                    meta['fps'] = float(num) / den_f
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        else:
            fps = _to_float(fps_raw)
            if fps:
                meta['fps'] = fps

        frames_raw = stream.get('nb_read_frames') or stream.get('nb_frames')
        if frames_raw not in (None, '', 'N/A'):
            try:
                meta['nb_frames'] = int(frames_raw)
            except (TypeError, ValueError):
                meta['nb_frames'] = None

        duration = _to_float(fmt.get('duration')) or _to_float(stream.get('duration'))
        if duration is None and meta['nb_frames'] and meta['fps'] and meta['fps'] > 0:
            duration = float(meta['nb_frames']) / float(meta['fps'])
        meta['duration'] = float(duration or 0.0)
        return meta
    except Exception as e:
        logger.error(f"Failed to probe video metadata for {video_path}: {e}")
        return meta


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe with fallbacks."""
    return float(_probe_video_metadata(video_path).get('duration') or 0.0)


def _run_ffmpeg_frame_extract(cmd: List[str], output_path: Path, label: str) -> bool:
    """Run one extraction strategy atomically and validate the produced PNG."""
    temp_output = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_output.exists():
        temp_output.unlink()
    full_cmd = list(cmd) + [str(temp_output)]
    logger.info(f"[SlidingStory] {label}: {' '.join(full_cmd)}")
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        if temp_output.exists() and temp_output.stat().st_size > 0:
            temp_output.replace(output_path)
            return True
        logger.warning(f"[SlidingStory] {label} produced no usable frame: {temp_output}")
        logger.debug(f"[SlidingStory] ffmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"[SlidingStory] {label} failed (rc={e.returncode}): {e.stderr}")
    finally:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
    return False


def _extract_last_frame_ffmpeg(video_path: Path, output_path: Path) -> None:
    """Extract the final decoded frame of a video using robust ffmpeg fallbacks.

    Why the belt-and-braces logic? Some generated H.264 clips can look fine during
    playback but produce blocky garbage when you jump near EOF with input-side `-ss`.
    We therefore prefer an exact frame-index decode when frame counts are known, and
    only then fall back to EOF seeking strategies.
    """
    if not video_path.exists():
        logger.error(f"[SlidingStory] Cannot extract last frame, video file does not exist: {video_path}")
        raise FileNotFoundError(f"Input video for frame extraction not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = _probe_video_metadata(video_path)
    duration = float(meta.get('duration') or 0.0)
    fps = float(meta.get('fps') or 0.0)
    nb_frames = meta.get('nb_frames')

    # Single-frame / ultra-short clips: the last frame is the first frame.
    if nb_frames == 1 or (duration > 0 and fps > 0 and duration <= max(0.35, 1.25 / fps)):
        logger.info(
            f"[SlidingStory] Last-frame extraction detected a single-frame or ultra-short clip "
            f"(frames={nb_frames}, duration={duration:.3f}s). Using first frame instead."
        )
        _extract_first_frame_ffmpeg(video_path, output_path)
        return

    strategies: List[tuple[List[str], str]] = []

    if isinstance(nb_frames, int) and nb_frames > 1:
        last_index = nb_frames - 1
        strategies.append(([
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f"select=eq(n\\,{last_index})",
            '-vsync', '0',
            '-frames:v', '1',
            '-q:v', '2',
        ], 'Extract last frame by exact frame index'))

    if duration > 0:
        for tail_offset in (0.001, 0.05, min(0.25, max(0.001, duration / 4.0))):
            if tail_offset >= duration:
                continue
            strategies.append(([
                'ffmpeg', '-y',
                '-sseof', f'-{tail_offset:.6f}',
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
            ], f'Extract last frame via EOF seek ({tail_offset:.3f}s)'))

    for cmd, label in strategies:
        if _run_ffmpeg_frame_extract(cmd, output_path, label):
            logger.debug(f"[SlidingStory] Last frame extracted → {output_path.name}")
            return

    logger.warning(
        f"[SlidingStory] All last-frame extraction strategies failed for {video_path}. "
        "Falling back to the first frame so the run can fail gracefully instead of writing a corrupted frame."
    )
    _extract_first_frame_ffmpeg(video_path, output_path)


def _extract_first_frame_ffmpeg(video_path: Path, output_path: Path) -> None:
    """Extract the first frame of a video using ffmpeg.

    Args:
        video_path: Path to the input MP4 video.
        output_path: Path to write the extracted PNG image.

    Raises:
        RuntimeError: If ffmpeg fails to extract the frame.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced empty output: {output_path}")
        logger.debug(f"[SlidingStory] First frame extracted → {output_path.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract first frame: {e.stderr}")


def _render_still_video_ffmpeg(image_path: Path, output_path: Path, *, duration_seconds: float, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-loop', '1',
        '-i', str(image_path),
        '-vf', f'fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p',
        '-t', f'{duration_seconds:.3f}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f'ffmpeg still fallback produced empty output: {output_path}\n{result.stderr}')


def _render_crossfade_video_ffmpeg(start_image: Path, end_image: Path, output_path: Path, *, duration_seconds: float, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fade_duration = max(0.15, min(duration_seconds * 0.75, max(0.15, duration_seconds - 0.05)))
    offset = max(0.0, duration_seconds - fade_duration)
    filter_graph = (
        f'[0:v]fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p[a];'
        f'[1:v]fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p[b];'
        f'[a][b]xfade=transition=fade:duration={fade_duration:.3f}:offset={offset:.3f},format=yuv420p[v]'
    )
    cmd = [
        'ffmpeg', '-y',
        '-loop', '1', '-t', f'{duration_seconds:.3f}', '-i', str(start_image),
        '-loop', '1', '-t', f'{duration_seconds:.3f}', '-i', str(end_image),
        '-filter_complex', filter_graph,
        '-map', '[v]',
        '-t', f'{duration_seconds:.3f}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f'ffmpeg crossfade fallback produced empty output: {output_path}\n{result.stderr}')


def _build_ffmpeg_continuity_fallback(*, request: GenerationRequest, conditioning_image: Path, last_frame_path: Optional[Path], require_morph: bool) -> Optional[GenerationResult]:
    duration_seconds = float(request.duration_seconds or 2.0)
    fps = float(request.video_fps or 4.0)
    output_path = request.output_dir / f'{request.atom_id}_video_ffmpeg_fallback.mp4'
    try:
        if require_morph and last_frame_path is not None and last_frame_path.exists():
            _render_crossfade_video_ffmpeg(last_frame_path, conditioning_image, output_path, duration_seconds=duration_seconds, fps=fps)
            mode = 'crossfade'
        else:
            _render_still_video_ffmpeg(conditioning_image, output_path, duration_seconds=duration_seconds, fps=fps)
            mode = 'still'
    except Exception as e:
        logger.warning(f'[SlidingStory] ffmpeg continuity fallback failed: {e}')
        return None

    logger.warning('[SlidingStory] Using ffmpeg %s fallback clip to avoid propagating a scrambled endpoint.', mode)
    return GenerationResult(success=True, video_path=output_path, metadata={'continuity_fallback': f'ffmpeg_{mode}'})


def _measure_video_endpoint_similarity(video_path: Path, target_image_path: Path) -> float:
    """Extract a temporary last frame and compare it with the conditioning target."""
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="vfaq_similarity_") as tmpdir:
        probe_frame = Path(tmpdir) / "lastframe_probe.png"
        _extract_last_frame_ffmpeg(video_path, probe_frame)
        with Image.open(probe_frame) as probe_img:
            probe_img.verify()
        with Image.open(target_image_path) as target_img:
            target_img.verify()
        return calculate_frame_similarity(str(probe_frame), str(target_image_path))


def _render_video_with_continuity_guard(
    *,
    backend,
    request: GenerationRequest,
    config: SlidingStoryConfig,
    require_morph: bool,
    conditioning_image: Path,
    last_frame_path: Optional[Path] = None,
) -> GenerationResult:
    """Render a conditioned video and retry with lower denoise if it drifts too hard."""
    attempts = max(0, int(config.continuity_retry_attempts)) + 1
    base_threshold = float(config.continuity_similarity_threshold)
    morph_threshold = float(getattr(config, 'continuity_morph_similarity_threshold', base_threshold))
    threshold = max(base_threshold, morph_threshold) if require_morph else base_threshold
    guard_enabled = bool(config.continuity_guard_enabled)
    force_fallback_for_morph = bool(getattr(config, 'continuity_force_fallback_for_morph', True)) and require_morph
    candidate_caps = (0.32, 0.24, 0.18, 0.12)

    base_denoise = float(request.denoise_strength)
    last_error: Optional[str] = None

    for attempt_idx in range(attempts):
        attempt_request = GenerationRequest(**request.__dict__)
        if attempt_idx > 0:
            cap = candidate_caps[min(attempt_idx - 1, len(candidate_caps) - 1)]
            attempt_request.denoise_strength = min(base_denoise, cap)
            logger.warning(
                "[SlidingStory] Continuity retry %s/%s with lower video denoise %.3f",
                attempt_idx,
                attempts - 1,
                attempt_request.denoise_strength,
            )

        if require_morph:
            result = backend.generate_morph_video(
                attempt_request,
                start_image_path=last_frame_path,
                end_image_path=conditioning_image,
            )
        else:
            result = backend.generate_video(attempt_request, source_image=conditioning_image)

        if not (result and result.success and result.video_path and result.video_path.exists()):
            last_error = getattr(result, 'error', 'unknown error')
            if attempt_idx + 1 >= attempts:
                return result
            logger.warning(
                "[SlidingStory] Video attempt %s/%s failed before continuity check: %s",
                attempt_idx + 1,
                attempts,
                last_error,
            )
            continue

        if not guard_enabled:
            return result

        try:
            similarity = _measure_video_endpoint_similarity(result.video_path, conditioning_image)
            result.metadata = dict(result.metadata or {})
            result.metadata['endpoint_similarity'] = similarity
            logger.info(
                "[SlidingStory] Video endpoint similarity vs conditioning image: %.3f (threshold=%.3f)",
                similarity,
                threshold,
            )
            if similarity >= threshold:
                return result

            if attempt_idx + 1 >= attempts:
                fallback_enabled = bool(config.continuity_ffmpeg_fallback_enabled)
                fallback_min_similarity = float(config.continuity_ffmpeg_fallback_min_similarity)
                should_fallback = fallback_enabled and (
                    force_fallback_for_morph or similarity < fallback_min_similarity
                )
                if should_fallback:
                    fallback_result = _build_ffmpeg_continuity_fallback(
                        request=attempt_request,
                        conditioning_image=conditioning_image,
                        last_frame_path=last_frame_path,
                        require_morph=require_morph,
                    )
                    if fallback_result is not None:
                        fallback_result.metadata = dict(fallback_result.metadata or {})
                        fallback_result.metadata['endpoint_similarity'] = similarity
                        fallback_result.metadata['replaced_video_path'] = str(result.video_path)
                        return fallback_result

                if force_fallback_for_morph:
                    return GenerationResult(
                        success=False,
                        error=(
                            f"Morph endpoint drifted too far from the target keyframe "
                            f"(similarity={similarity:.3f} < threshold={threshold:.3f}) and no fallback clip could be produced."
                        ),
                        video_path=result.video_path,
                        metadata=result.metadata,
                    )

                logger.warning(
                    "[SlidingStory] Continuity guard accepted a weak result after exhausting retries "
                    "(similarity=%.3f < %.3f).",
                    similarity,
                    threshold,
                )
                return result
        except Exception as e:
            logger.warning(f"[SlidingStory] Continuity guard failed; keeping generated video: {e}")
            return result

        logger.warning(
            "[SlidingStory] Endpoint drift too high (similarity %.3f < %.3f). Retrying video generation.",
            similarity,
            threshold,
        )

    return GenerationResult(success=False, error=last_error or "Continuity-guarded video generation failed")


def run_sliding_story(
    story_path: Path,
    qodeyard_dir: Path,
    config: SlidingStoryConfig,
    max_cycles: Optional[int] = None,
    base_image_path: Optional[Path] = None,
    base_video_path: Optional[Path] = None,
    # ── Resume parameters (v0.6.1-beta) ───────────────────────────────────
    start_cycle: int = 1,
    initial_last_frame_path: Optional[Path] = None,
    initial_anchor_frame_path: Optional[Path] = None,
    initial_final_video_paths: Optional[List[Path]] = None,
    initial_completed_cycles: Optional[set] = None,
    checkpoint_callback: Optional[callable] = None,
) -> Path:
    """Execute the sliding window story engine.

    This function orchestrates the entire run: parsing the story,
    determining the generation schedule, invoking the backend for
    keyframe and img2img generation, generating cross‑fade videos, and
    assembling the final video. All intermediate and final artifacts are
    stored under `qodeyard_dir`.

    Args:
        story_path: Path to the input story text file.
        qodeyard_dir: Directory where all outputs are written.
        config: SlidingStoryConfig with runtime parameters.
        max_cycles: Optional override for maximum number of cycles
                    (e.g. from audio sync).
        base_image_path: Optional base image for image/video mode
                         (used as cycle 0 start image instead of txt2img).
        base_video_path: Optional base video for Veo extend_video bootstrap
                         (v0.6.0-beta). When set and backend is Veo with
                         extension enabled, cycle 1 uses this as extend input
                         instead of text_to_video or image_to_video.
        start_cycle: Cycle index to start from (1-based). For resume, set
                     to the first incomplete cycle.
        initial_last_frame_path: Pre-existing last frame for resume.
        initial_anchor_frame_path: Pre-existing anchor frame for resume.
        initial_final_video_paths: Pre-existing ordered video list for resume.
        initial_completed_cycles: Set of already-completed cycle indices.
        checkpoint_callback: Called after each successful cycle with
                             (cycle_idx, last_frame_path, video_path, anchor_path).

    Returns:
        Path to the final concatenated story video.
    """
    logger.info(f"[SlidingStory] Starting story run. Reinject: {config.reinject}. Story: {story_path.name}")
    # Ensure output directory structure exists
    qodeyard_dir = qodeyard_dir.resolve()
    keyframes_dir = qodeyard_dir / "frames"
    lastframes_dir = qodeyard_dir / "frames"
    videos_dir = qodeyard_dir / "videos"
    briqs_dir = qodeyard_dir / "briqs"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    briqs_dir.mkdir(parents=True, exist_ok=True)

    # Backend temp output directory (ComfyUI writes here; we copy artifacts into run/)
    # Keep it inside qodeyard_dir so everything stays on the same filesystem.
    comfy_tmp = Path(tempfile.mkdtemp(prefix="backend_tmp_", dir=str(qodeyard_dir)))

    # Copy story file into qodeyard
    dest_story = qodeyard_dir / "story.txt"
    shutil.copyfile(story_path, dest_story)

    # Parse paragraphs
    paragraphs = _parse_story_file(story_path)
    if not paragraphs:
        raise RuntimeError("Story contains no paragraphs; nothing to generate")
    P = len(paragraphs)
    M = config.max_paragraphs
    windows = _determine_windows(P, M)

    motion_prompt_text = _load_optional_prompt_text(story_path.parent, "motion_prompt.md")
    if motion_prompt_text:
        logger.info(f"[SlidingStory] Loaded: motion_prompt.md ({len(motion_prompt_text)} chars)")

    # Apply max_cycles override if specified (e.g. from audio sync)
    if max_cycles is not None and max_cycles > 0 and max_cycles < len(windows):
        windows = windows[:max_cycles]
        logger.info(f"[SlidingStory] Capped to {max_cycles} cycles (audio sync)")

    total_cycles = len(windows)
    logger.info(
        f"[SlidingStory] Parsed {P} paragraphs → {total_cycles} cycles "
        f"with window size ≤ {M}"
    )

    # Instantiate backend
    backend_cfg = config.backend_config or {}
    backend = create_backend(backend_cfg)

    backend_summary = describe_backend_config(backend_cfg)
    image_backend_type = get_backend_type_for_capability(backend_cfg, 'image')
    video_backend_type = get_backend_type_for_capability(backend_cfg, 'video')
    morph_backend_type = get_backend_type_for_capability(backend_cfg, 'morph')

    # ── Capability-aware backend detection ────────────────────────────────
    is_veo = video_backend_type == 'veo'

    if is_veo:
        logger.info("[SlidingStory] Veo video backend detected — using direct video generation pipeline")
    is_venice = video_backend_type == 'venice'

    morph_is_veo = morph_backend_type == 'veo'
    morph_is_ltx = morph_backend_type == 'ltx_video'
    morph_is_venice = morph_backend_type == 'venice'
    resolved_backend_cfgs = resolve_capability_backend_configs(backend_cfg)
    smart_reinject_requested = bool(getattr(config, "smart_reinject_enabled", False))
    smart_reinject_state = _SmartReinjectState(enabled=bool(is_venice and smart_reinject_requested))
    if smart_reinject_requested and not is_venice:
        logger.info(
            "[SlidingStory/SmartReinject] smart_reinject_enabled=true but video backend is not Venice; "
            "feature is currently Venice-first and will be ignored."
        )
    if smart_reinject_state.enabled:
        try:
            image_cap_cfg = resolve_capability_backend_configs(backend_cfg).get("image", {})
            smart_reinject_state.image_backend = create_backend(image_cap_cfg or {})
            smart_reinject_state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart_reinject")
            _register_smart_reinject_state(smart_reinject_state)
            logger.info("[SlidingStory/SmartReinject] Async prefetch enabled.")
        except Exception as e:
            smart_reinject_state.enabled = False
            smart_reinject_state.executor = None
            smart_reinject_state.image_backend = None
            logger.warning(f"[SlidingStory/SmartReinject] Failed to initialize async prefetch backend: {e}")

    def get_capability_setting(capability: str, key: str, default=None):
        cap_cfg = resolved_backend_cfgs.get(capability, {})
        if isinstance(cap_cfg, dict) and cap_cfg.get(key) is not None:
            return cap_cfg.get(key)
        return backend_cfg.get(key, default)

    # Anchor frame for loop closure (first cycle's first frame)
    anchor_frame_path: Optional[Path] = None
    # Track which cycles were Veo-generated (for extension eligibility)
    veo_generated_cycles: set = set()

    # Instantiate Finalizer and get per-cycle interpolation settings
    finalizer_cfg = config.finalizer_config or {}
    finalizer = Finalizer(
        project_dir=qodeyard_dir,
        finalizer_config=finalizer_cfg
    )
    per_cycle_interpolation_enabled = finalizer_cfg.get('per_cycle_interpolation', False)
    # v0.9.1: pingpong is now independent of interpolation. The flag controls
    # whether _process_cycle_video should be called at all (either stage on
    # is enough), AND whether the LAST FRAME for the next cycle's morph must
    # be extracted from the ORIGINAL (pre-pingpong) video — because the
    # pingpong'd video's actual last frame == its first frame, which would
    # make subsequent cycles loop on themselves.
    per_cycle_pingpong_enabled = finalizer_cfg.get('per_cycle_pingpong', False)
    per_cycle_processing_enabled = per_cycle_interpolation_enabled or per_cycle_pingpong_enabled
    per_cycle_interpolate_fps = finalizer_cfg.get('interpolate_fps', 30) # Default to 30 as per spec

    # Resolve timing parameters
    resolver = TimingResolver()
    
    # ── Veo timing override ───────────────────────────────────────────────
    # Veo controls its own fps/duration natively. We still resolve timing
    # for the Finalizer and for metadata, but the authoritative duration
    # for Veo comes from the veo: config section.
    resolver_fps = config.video_fps
    venice_duration = config.img2vid_duration_sec
    veo_duration = None
    if is_veo:
        veo_cfg_dict = config.veo_config or backend_cfg.get('veo', {})
        veo_duration = veo_cfg_dict.get('duration_seconds', 8)
        logger.info(f"[SlidingStory] Veo duration authority: {veo_duration}s")
    resolver_frames = config.video_frames
    resolver_duration = config.img2vid_duration_sec
    resolver_authority = config.timing_authority

    # If no explicit authority is given, assume 'fps' as default and provide fallback values
    if resolver_authority is None:
        resolver_authority = "fps" # Assume 'fps' is the default authority
        # If fps is also missing, provide a default
        if resolver_fps is None:
            resolver_fps = 8.0
        # If both frames and duration are missing, provide a default duration
        if resolver_duration is None and resolver_frames is None:
            resolver_duration = 3.0
        # If only frames is provided, resolver can calculate duration
        # If only duration is provided, resolver can calculate frames
    
    # If a specific authority is provided, ensure corresponding critical values are present
    # These checks are now more about guiding the user if they provide an authority but no data
    if resolver_authority == "frames" and resolver_frames is None:
        raise ValueError("When timing_authority is 'frames', video_frames must be provided in config.")
    if resolver_authority == "duration" and resolver_duration is None:
        raise ValueError("When timing_authority is 'duration', img2vid_duration_sec must be provided in config.")
    if resolver_authority == "fps" and resolver_fps is None:
        raise ValueError("When timing_authority is 'fps', video_fps must be provided in config.")

    # Call the resolver with the prepared (potentially defaulted) values
    resolved_timing = resolver.resolve(
        fps=resolver_fps,
        frames=resolver_frames,
        duration=resolver_duration,
        authority=resolver_authority
    )
    resolved_fps = resolved_timing["resolved_fps"]
    resolved_frames = resolved_timing["resolved_frames"]
    resolved_duration = resolved_timing["resolved_duration"]


    # Track last frame path for chaining
    last_frame_path: Optional[Path] = initial_last_frame_path
    final_video_paths: List[Path] = list(initial_final_video_paths or [])

    # Seed management for determinism: derive cycle‑specific seeds
    random_seed_base = config.seed_base

    # ── Resume state (v0.6.1-beta) ────────────────────────────────────────
    # Restore anchor frame and Veo-generated tracking from resume context
    if initial_anchor_frame_path:
        anchor_frame_path = initial_anchor_frame_path
        logger.info(f"[Resume] Restored anchor frame: {anchor_frame_path}")
    if initial_completed_cycles:
        veo_generated_cycles = set(initial_completed_cycles)
        logger.info(f"[Resume] Restored {len(veo_generated_cycles)} completed cycle(s)")

    resuming = start_cycle > 1
    if resuming:
        logger.info(
            f"[Resume] Resuming from cycle {start_cycle} "
            f"(last_frame={'yes' if last_frame_path else 'no'}, "
            f"videos={len(final_video_paths)}, anchor={'yes' if anchor_frame_path else 'no'})"
        )

    # Initialize Crowd Control client (fail-open: errors return None)
    crowd_client = None
    crowd_cc_cfg = config.crowd_control_config or {}
    crowd_carryover_cycles = 0
    if crowd_cc_cfg.get("enabled", False):
        try:
            from vfaq.crowd_control.models import CrowdControlConfig
            from vfaq.crowd_control.client import CrowdClient
            cc_config = CrowdControlConfig.from_dict(crowd_cc_cfg)
            crowd_client = CrowdClient(cc_config)
            crowd_inject_mode = cc_config.inject_mode
            crowd_inject_source_mode = cc_config.inject_source_mode
            crowd_inject_label = cc_config.inject_label
            crowd_carryover_cycles = _normalize_crowd_carryover_cycles(
                getattr(cc_config, "carryover_cycles", 0)
            )
            if "inject_source_mode" not in crowd_cc_cfg:
                logger.info(
                    "[SlidingStory] crowd_control.inject_source_mode not explicitly set; "
                    f"defaulting to '{crowd_inject_source_mode}'."
                )
            logger.info(
                f"[SlidingStory] Crowd Control enabled — inject_mode={crowd_inject_mode}, "
                f"inject_source_mode={crowd_inject_source_mode}, carryover_cycles={crowd_carryover_cycles}"
            )
        except Exception as e:
            logger.warning(f"[SlidingStory] Crowd Control init failed (continuing without): {e}")
            crowd_client = None
            crowd_inject_mode = "append"
            crowd_inject_source_mode = "as_image_source"
            crowd_inject_label = "Audience mutation request"
            crowd_carryover_cycles = 0
    else:
        crowd_inject_mode = "append"
        crowd_inject_source_mode = "as_image_source"
        crowd_inject_label = "Audience mutation request"
        crowd_carryover_cycles = 0

    crowd_carryover_prompt: Optional[str] = None
    crowd_carryover_remaining = 0
    for cycle_idx, window_indices in enumerate(windows, start=1):
        _reap_abandoned_smart_reinject_prefetches(smart_reinject_state)
        # ── Resume: skip already-completed cycles ─────────────────────────
        if cycle_idx < start_cycle:
            continue

        logger.info(f"[SlidingStory] Cycle \033[92m{cycle_idx}\033[0m/{total_cycles} — window paragraphs {window_indices}")
        # Concatenate paragraphs as the base story prompt; crowd resolution
        # happens once via ResolvedCyclePrompt so all downstream paths use the
        # same authoritative prompt object.
        base_story_prompt = "\n\n".join(paragraphs[i - 1] for i in window_indices)
        logger.debug(f"[SlidingStory] Stacked prompt:\n{base_story_prompt}")

        # ── Crowd Control: check queue and inject if available ───────────
        crowd_prompt_used: Optional[str] = None
        crowd_prompt_origin = "none"
        if crowd_client is not None:
            try:
                fresh_prompt = crowd_client.pop_next()
                if fresh_prompt:
                    crowd_prompt_used = str(fresh_prompt).strip()
                    crowd_prompt_origin = "fresh"
                    crowd_carryover_prompt = crowd_prompt_used if crowd_carryover_cycles > 0 else None
                    crowd_carryover_remaining = crowd_carryover_cycles
                elif crowd_carryover_prompt and crowd_carryover_remaining > 0:
                    crowd_prompt_used = crowd_carryover_prompt
                    crowd_prompt_origin = "carryover"
                    crowd_carryover_remaining = max(0, crowd_carryover_remaining - 1)
                    if crowd_carryover_remaining <= 0:
                        crowd_carryover_prompt = None
            except Exception as e:
                logger.warning(f"[SlidingStory] Crowd Control error (fail-open): {e}")
                crowd_prompt_used = None
                crowd_prompt_origin = "none"

        resolved_cycle_prompt = _resolve_cycle_prompt(
            base_story_prompt=base_story_prompt,
            crowd_prompt=crowd_prompt_used,
            crowd_prompt_origin=crowd_prompt_origin,
            inject_mode=crowd_inject_mode,
            inject_source_mode=crowd_inject_source_mode,
            inject_label=crowd_inject_label,
        )
        stacked_prompt = resolved_cycle_prompt.resolved_main_prompt

        video_stage_prompt = _build_video_stage_prompt(stacked_prompt, motion_prompt_text)

        # ── Crowd-driven cycle flow flags (v0.9.1) ────────────────────────
        # The crowd-injected cycle is driven by inject_source_mode:
        #
        #   "as_image_source"  → IMG2VID with lastframe as init/source image.
        #     The audience prompt drives motion/content; the lastframe gives
        #     hard visual continuity from the previous cycle. This is what
        #     the user wants in spec examples 1 and 2.
        #
        #   "as_reference"     → TEXT2VID with lastframe as a reference image
        #     (when the model supports reference_image_urls). The audience
        #     prompt is the primary driver; the lastframe becomes a soft
        #     style/identity tether. This is what the user wants in spec
        #     examples 3 and 4.
        #
        # `inject_mode` (append vs replace) is now ORTHOGONAL to source mode
        # and only affects WHAT TEXT goes into the prompt — story+crowd vs
        # crowd-only. It NO LONGER forces a text2img visual reset.
        crowd_active = bool(resolved_cycle_prompt.crowd_active)
        crowd_use_reference_mode = (
            crowd_active and resolved_cycle_prompt.inject_source_mode == "as_reference"
        )
        smart_reinject_schedule_allowed_for_cycle = bool(
            smart_reinject_state.enabled and not crowd_active
        )
        smart_reinject_consume_allowed_for_cycle = bool(
            smart_reinject_state.enabled and not crowd_active
        )
        # Reinject (img2img keyframe) is suppressed during a crowd-driven
        # cycle in EITHER source mode — the audience prompt is meant to
        # drive the new cycle, not be diluted by an img2img remix of the
        # prior frame. Morph is similarly suppressed: the goal is mutation,
        # not seamless transition.
        effective_reinject = config.reinject and not crowd_active
        effective_require_morph = config.require_morph and effective_reinject
        if crowd_active:
            logger.info(
                f"[SlidingStory] Crowd-driven cycle: source_mode={resolved_cycle_prompt.inject_source_mode}, "
                f"inject_mode={resolved_cycle_prompt.inject_mode}, origin={resolved_cycle_prompt.crowd_prompt_origin} "
                f"(reinject + morph suppressed for this cycle)"
            )
            if smart_reinject_state.enabled:
                logger.info(
                    "[SlidingStory/SmartReinject] Crowd prompt active: smart reinject "
                    "scheduling/consumption/application disabled for this cycle."
                )
        # Legacy alias kept for downstream code paths that still check it,
        # but its meaning is narrowed: True only in the old "replace + reset"
        # combination, which now requires both crowd_active AND the
        # as_reference mode (the only path that ditches the lastframe as
        # source). For as_image_source we keep using img2vid with lastframe.
        crowd_replace_active = crowd_use_reference_mode
        if crowd_use_reference_mode:
            logger.info(
                "[SlidingStory] Crowd cycle (as_reference): TEXT2VID with previous "
                "lastframe as REFERENCE image"
            )

        # Determine unique atom_id for this cycle
        atom_id = f"cycle_{cycle_idx:03d}"

        # Determine seed for this cycle (offset by cycle index)
        seed = random_seed_base + cycle_idx

        # Briq data for this cycle
        briq_data = {
            "cycle_index": cycle_idx,
            "paragraph_window": {"start": window_indices[0], "end": window_indices[-1]},
            "paragraph_text": base_story_prompt[:500],
            "resolved_main_prompt_preview": stacked_prompt[:500],
            "seed": seed,
            "input_mode": "text",
            "reinject": config.reinject,
            "backend_type": backend_summary,
            "image_backend_type": image_backend_type,
            "video_backend_type": video_backend_type,
            "morph_backend_type": morph_backend_type,
            "video_prompt_preview": video_stage_prompt[:500],
            "motion_prompt_used": bool(motion_prompt_text),
            "crowd_active": crowd_active,
            "crowd_prompt_used": resolved_cycle_prompt.crowd_prompt,
            "crowd_prompt_origin": resolved_cycle_prompt.crowd_prompt_origin,
            "crowd_carryover_remaining": crowd_carryover_remaining,
            "requested_inject_mode": resolved_cycle_prompt.inject_mode,
            "requested_inject_source_mode": resolved_cycle_prompt.inject_source_mode,
            "authoritative_prompt_route": "crowd_main_route" if crowd_active else "story_main_route",
            "smart_reinject_enabled": bool(smart_reinject_state.enabled),
            "smart_reinject_schedule_allowed": smart_reinject_schedule_allowed_for_cycle,
            "smart_reinject_consume_allowed": smart_reinject_consume_allowed_for_cycle,
            "smart_reinject_apply_allowed": smart_reinject_consume_allowed_for_cycle,
            "smart_reinject_pending_detected": False,
            "smart_reinject_discarded_due_to_crowd": False,
            "smart_reinject_missed_target_cycle": False,
            "smart_reinject_paused_due_to_crowd": bool(crowd_active and smart_reinject_state.enabled),
            "smart_reinject_skip_reason": (
                "crowd_prompt_active" if crowd_active and smart_reinject_state.enabled else None
            ),
            "smart_reinject_used": False,
            "smart_reinject_similarity": None,
        }
        if smart_reinject_state.enabled:
            briq_data.update({
                "smart_reinject_keyframe": None,
                "smart_reinject_source_cycle": None,
            })

        # Record crowd control state in briq (includes visual reset telemetry)
        if crowd_client is not None:
            if resolved_cycle_prompt.crowd_prompt:
                briq_data["crowd_control"] = {
                    "used": True,
                    "prompt_preview": resolved_cycle_prompt.crowd_prompt[:120],
                    "prompt_origin": resolved_cycle_prompt.crowd_prompt_origin,
                    "inject_mode": resolved_cycle_prompt.inject_mode,
                    "inject_source_mode": resolved_cycle_prompt.inject_source_mode,
                    "visual_reset": crowd_replace_active,
                    "effective_reinject": effective_reinject,
                    "carryover_remaining": crowd_carryover_remaining,
                }
            else:
                briq_data["crowd_control"] = {
                    "used": False,
                    "prompt_origin": "none",
                    "inject_mode": resolved_cycle_prompt.inject_mode,
                    "inject_source_mode": resolved_cycle_prompt.inject_source_mode,
                    "visual_reset": False,
                    "effective_reinject": effective_reinject,
                    "carryover_remaining": crowd_carryover_remaining,
                }

        # ═══════════════════════════════════════════════════════════════════
        # VEO BACKEND CYCLE PATH (v0.6.0-beta)
        # ═══════════════════════════════════════════════════════════════════
        # Veo generates video directly — no separate txt2img/img2img step.
        # Orchestration params (continuity/mutation/identity strengths) from
        # the veo: config drive the scheduling decisions here.
        if is_veo:
            veo_mode_str = "text_to_video"
            image_for_veo = None
            last_frame_for_veo = None
            ref_image_paths = []
            ref_image_types = []

            # ── Resolve orchestration params from veo config ──────────────
            veo_orch = config.veo_config or backend_cfg.get('veo', {})
            orch_continuity = float(veo_orch.get('continuity_strength', 0.85))
            orch_mutation = float(veo_orch.get('mutation_strength', 0.25))
            orch_identity = float(veo_orch.get('identity_lock_strength', 0.80))
            orch_loop = float(veo_orch.get('loop_closure_strength', 0.90))
            veo_enable_ext = bool(veo_orch.get('enable_extension', False))
            veo_enable_refs = bool(veo_orch.get('enable_reference_images', True))

            # ── Map continuity/mutation to effective denoise range ─────────
            # High continuity + low mutation → narrow low denoise range (mild evolution)
            # Low continuity + high mutation → wide high denoise range (strong evolution)
            eff_denoise_min = max(0.05, config.img2img_denoise_min * (1.0 - orch_continuity + orch_mutation))
            eff_denoise_max = min(0.95, config.img2img_denoise_max * (1.0 + orch_mutation - orch_continuity * 0.5))
            if eff_denoise_min > eff_denoise_max:
                eff_denoise_min, eff_denoise_max = eff_denoise_max, eff_denoise_min

            # ── Collect reference images for identity lock ────────────────
            # When identity_lock_strength > 0.5 and we have anchor/keyframes,
            # feed them as reference images to maintain subject consistency.
            # Veo 3.1 uses ASSET type (STYLE is only for veo-2.0-generate-exp).
            if veo_enable_refs and orch_identity > 0.5 and cycle_idx > 1:
                veo_model_id = veo_orch.get('model', 'veo-3.1-generate-preview')
                # Model-aware reference type: STYLE only for veo-2.0-exp, ASSET for 3.x
                if 'veo-2.0' in veo_model_id and 'exp' in veo_model_id:
                    ref_type = "STYLE"
                else:
                    ref_type = "ASSET"

                if anchor_frame_path and anchor_frame_path.exists():
                    ref_image_paths.append(anchor_frame_path)
                    ref_image_types.append(ref_type)
                # Also add previous keyframe if it exists (most recent visual identity)
                prev_kf = keyframes_dir / f"keyframe_{cycle_idx - 1:03d}.png"
                if prev_kf.exists() and len(ref_image_paths) < 3:
                    ref_image_paths.append(prev_kf)
                    ref_image_types.append(ref_type)
                if ref_image_paths:
                    logger.info(
                        f"[SlidingStory/Veo] Identity lock: {len(ref_image_paths)} "
                        f"reference image(s) type={ref_type} (strength={orch_identity:.2f}, "
                        f"model={veo_model_id})"
                    )

            # ── Determine Veo mode for this cycle ─────────────────────────
            # default_mode from veo config is used when no explicit signal
            # (base image, last frame, morph) determines the mode.
            veo_default_mode = veo_orch.get('default_mode', 'image_to_video')

            if cycle_idx == 1:
                # ── Veo cycle 1: check for base_video extension bootstrap ─
                # If a base video is provided, extension is enabled, and we're
                # on Vertex, use extend_video to continue from user-supplied video.
                veo_provider = veo_orch.get('provider', 'gemini').lower()
                base_vid_ok = (
                    base_video_path
                    and base_video_path.exists()
                    and veo_enable_ext
                    and veo_provider == 'vertex'
                    and not crowd_replace_active
                )
                if base_vid_ok:
                    base_vid_dur = get_video_duration(base_video_path)
                    if 1.0 <= base_vid_dur <= 30.0:
                        veo_mode_str = "extend_video"
                        briq_data["input_mode"] = "extend_bootstrap"
                        logger.info(
                            f"[SlidingStory/Veo] Cycle 1: extend_video from base video "
                            f"({base_video_path.name}, {base_vid_dur:.1f}s)"
                        )
                        # Copy base video to videos dir as "cycle 0" source
                        bootstrap_src = videos_dir / "video_000_bootstrap.mp4"
                        shutil.copyfile(base_video_path, bootstrap_src)
                    else:
                        logger.warning(
                            f"[SlidingStory/Veo] Base video {base_vid_dur:.1f}s outside "
                            f"1–30s for extension — falling back to image/text mode"
                        )
                        base_vid_ok = False

                if not base_vid_ok and base_image_path and base_image_path.exists() and not crowd_replace_active:
                    veo_mode_str = "image_to_video"
                    image_for_veo = base_image_path
                    briq_data["input_mode"] = "image"
                    logger.info(f"[SlidingStory/Veo] Cycle 1: image_to_video from base image")
                elif not base_vid_ok:
                    # No base image — use text_to_video (default_mode doesn't
                    # override here since there's no image to use for img2vid)
                    briq_data["input_mode"] = "text"
                    logger.info(f"[SlidingStory/Veo] Cycle 1: text_to_video")
            else:
                # ── Check for video extension eligibility ─────────────────
                # Extension requires ALL of:
                # 1. enable_extension=true in config
                # 2. provider=vertex (extension is Vertex-only in Veo SDK)
                # 3. Previous cycle was Veo-generated (tracked via veo_generated_cycles set)
                # 4. Previous output video exists and is 1–30s long
                # 5. Not in morph mode and not crowd-replaced
                prev_video = videos_dir / f"video_{cycle_idx - 1:03d}.mp4"
                veo_provider = veo_orch.get('provider', 'gemini').lower()
                can_extend = (
                    veo_enable_ext
                    and veo_provider == 'vertex'
                    and not effective_require_morph
                    and not crowd_replace_active
                    and prev_video.exists()
                    and (cycle_idx - 1) in veo_generated_cycles
                )
                if can_extend:
                    # Validate input duration (Veo extend requires 1–30s input)
                    prev_dur = get_video_duration(prev_video)
                    if 1.0 <= prev_dur <= 30.0:
                        veo_mode_str = "extend_video"
                        briq_data["input_mode"] = "extend"
                        logger.info(
                            f"[SlidingStory/Veo] Cycle {cycle_idx}: extend_video from "
                            f"previous clip ({prev_dur:.1f}s, Vertex)"
                        )
                    else:
                        logger.warning(
                            f"[SlidingStory/Veo] Previous clip duration {prev_dur:.1f}s "
                            f"outside 1–30s range for extension — falling back to image_to_video"
                        )
                        can_extend = False
                elif veo_enable_ext and veo_provider != 'vertex' and not effective_require_morph:
                    logger.info(
                        f"[SlidingStory/Veo] Extension skipped: requires provider=vertex "
                        f"(current: {veo_provider})"
                    )

                if not can_extend and effective_require_morph and last_frame_path and last_frame_path.exists():
                    # ── Morph mode: evolved target keyframe + first_last_frame ─
                    rng = random.Random(random_seed_base + cycle_idx * 10007)
                    denoise = rng.uniform(eff_denoise_min, eff_denoise_max)
                    briq_data["denoise"] = denoise

                    logger.info(
                        f"[SlidingStory/Veo] Cycle {cycle_idx}: morph mode "
                        f"(denoise={denoise:.3f}, continuity={orch_continuity:.2f})"
                    )

                    kf_req = GenerationRequest(
                        prompt=stacked_prompt,
                        negative_prompt="",
                        seed=seed,
                        mode=InputMode.IMAGE,
                        init_image_path=last_frame_path,
                        denoise_strength=denoise,
                        width=get_capability_setting('image', 'width', 1280),
                        height=get_capability_setting('image', 'height', 720),
                        output_dir=keyframes_dir,
                        atom_id=f"{atom_id}_kf",
                    )
                    kf_result = backend.generate_image(kf_req)
                    if kf_result.success and kf_result.image_path:
                        keyframe_name = f"keyframe_{cycle_idx:03d}.png"
                        keyframe_path = keyframes_dir / keyframe_name
                        if kf_result.image_path != keyframe_path:
                            shutil.move(str(kf_result.image_path), str(keyframe_path))
                        logger.info(f"[SlidingStory/Veo] Evolved target keyframe → {keyframe_path}")
                        veo_mode_str = "first_last_frame"
                        image_for_veo = last_frame_path
                        last_frame_for_veo = keyframe_path
                        briq_data["paths"] = {
                            "keyframe": str(keyframe_path),
                            "start_image": str(last_frame_path),
                        }
                    else:
                        logger.warning("[SlidingStory/Veo] Evolved keyframe failed, falling back to image_to_video")
                        veo_mode_str = "image_to_video"
                        image_for_veo = last_frame_path

                elif last_frame_path and last_frame_path.exists():
                    # Respect default_mode: if user explicitly wants text_to_video
                    # for every cycle, skip last-frame chaining.
                    if veo_default_mode == "text_to_video":
                        veo_mode_str = "text_to_video"
                        briq_data["input_mode"] = "text"
                        logger.info(
                            f"[SlidingStory/Veo] Cycle {cycle_idx}: text_to_video "
                            f"(default_mode override, ignoring last frame)"
                        )
                    else:
                        veo_mode_str = "image_to_video"
                        image_for_veo = last_frame_path
                        briq_data["input_mode"] = "image"
                        logger.info(f"[SlidingStory/Veo] Cycle {cycle_idx}: image_to_video from last frame")
                else:
                    veo_mode_str = "text_to_video"
                    briq_data["input_mode"] = "text"
                    logger.info(f"[SlidingStory/Veo] Cycle {cycle_idx}: text_to_video (no prior frame)")

            # ── Build and dispatch Veo generation request ─────────────────
            # Veo outputs directly to run/videos/ — no temp dir needed.
            vid_atom_id = f"video_{cycle_idx:03d}"
            vid_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.IMAGE if image_for_veo else InputMode.TEXT,
                init_image_path=image_for_veo,
                width=get_capability_setting('video', 'width', 1280),
                height=get_capability_setting('video', 'height', 720),
                output_dir=videos_dir,
                atom_id=vid_atom_id,
                duration_seconds=veo_duration,
                video_fps=resolved_fps,
                video_frames=resolved_frames,
                video_prompt=video_stage_prompt,
                motion_prompt=motion_prompt_text or None,
                veo_mode=veo_mode_str,
                last_frame_path=last_frame_for_veo,
                reference_image_paths=ref_image_paths if ref_image_paths else None,
                reference_image_types=ref_image_types if ref_image_types else None,
            )

            # Dispatch by mode
            vid_result = None
            if veo_mode_str == "first_last_frame" and image_for_veo and last_frame_for_veo:
                vid_result = backend.generate_morph_video(
                    vid_req,
                    start_image_path=image_for_veo,
                    end_image_path=last_frame_for_veo,
                )
            elif veo_mode_str == "extend_video":
                # Determine extension source: bootstrap video for cycle 1,
                # or previous cycle's output for subsequent cycles.
                if cycle_idx == 1:
                    ext_source = videos_dir / "video_000_bootstrap.mp4"
                else:
                    ext_source = videos_dir / f"video_{cycle_idx - 1:03d}.mp4"
                vid_result = backend.generate_extension(vid_req, ext_source)
            else:
                vid_result = backend.generate_video(
                    vid_req,
                    source_image=image_for_veo,  # None for text_to_video
                )

            if not (vid_result and vid_result.success and vid_result.video_path and vid_result.video_path.exists()):
                error_msg = getattr(vid_result, 'error', 'unknown error')
                raise RuntimeError(f"[Veo] Video generation failed in cycle {cycle_idx}: {error_msg}")

            # Veo outputs as <atom_id>_veo.mp4 — rename to standard video_NNN.mp4
            video_name = f"video_{cycle_idx:03d}.mp4"
            video_path = videos_dir / video_name
            if vid_result.video_path != video_path:
                shutil.move(str(vid_result.video_path), str(video_path))
            logger.info(f"[SlidingStory/Veo] Generated video → {video_path}")

            current_cycle_video_path = video_path

            # Per-cycle interpolation/pingpong (shared with ComfyUI path).
            # v0.9.1: also runs when only pingpong is enabled.
            if per_cycle_processing_enabled:
                interpolated_video_path = finalizer._process_cycle_video(
                    video_path, resolved_fps
                )
                if interpolated_video_path:
                    current_cycle_video_path = interpolated_video_path

            # Extract last frame.
            # When pingpong is on, current_cycle_video_path's actual last frame
            # equals its FIRST frame (it ran forward then backward), which would
            # make the next cycle visually loop. Extract from the original
            # pre-pingpong video instead — that frame is the true "end of new
            # visual content" for this cycle.
            lastframe_source = (
                video_path if per_cycle_pingpong_enabled else current_cycle_video_path
            )
            lastframe_name = f"lastframe_{cycle_idx:03d}.png"
            lastframe_path = keyframes_dir / lastframe_name
            _extract_last_frame_ffmpeg(lastframe_source, lastframe_path)
            last_frame_path = lastframe_path
            final_video_paths.append(current_cycle_video_path)

            # Save anchor frame from cycle 1 for loop closure + identity refs
            if cycle_idx == 1:
                anchor_frame_path = keyframes_dir / "anchor_frame_001.png"
                _extract_first_frame_ffmpeg(current_cycle_video_path, anchor_frame_path)
                logger.debug(f"[SlidingStory] Anchor frame saved → {anchor_frame_path.name}")

            briq_data.setdefault("paths", {})
            briq_data["paths"]["video"] = str(current_cycle_video_path)
            briq_data["paths"]["last_frame"] = str(lastframe_path)
            briq_data["veo_mode"] = veo_mode_str
            briq_data["orchestration"] = {
                "continuity_strength": orch_continuity,
                "mutation_strength": orch_mutation,
                "identity_lock_strength": orch_identity,
                "reference_images_used": len(ref_image_paths),
            }
            if vid_result.metadata:
                briq_data["veo_metadata"] = vid_result.metadata
            # Track this cycle as Veo-generated for extension eligibility
            veo_generated_cycles.add(cycle_idx)
            _write_briq_json(briqs_dir, cycle_idx, briq_data)
            # ── Checkpoint callback (v0.6.1-beta) ────────────────────────
            if checkpoint_callback:
                checkpoint_callback(cycle_idx, last_frame_path, current_cycle_video_path, anchor_frame_path)
            continue  # Skip ComfyUI cycle logic below

        # ═══════════════════════════════════════════════════════════════════

        # VENICE BACKEND CYCLE PATH (v0.7.1-beta)
        # ═══════════════════════════════════════════════════════════════════
        # Venice can do text2vid and img2vid natively. We therefore route it
        # like Veo/LTX instead of forcing the classic keyframe-first SVD path.
        if is_venice:
            venice_mode_str = "text_to_video"
            image_for_venice = None
            end_image_for_venice = None
            venice_cfg_dict = config.venice_config or backend_cfg.get('venice', {})
            venice_prefers_t2v = bool(venice_cfg_dict.get('text_to_video_first_cycle', True))

            # ── Per-op duration resolution ────────────────────────────────
            # Engine reads per-op sub-blocks directly so the correct duration
            # is passed as request.duration_seconds (which the backend selector
            # checks first). This ensures text2vid and img2vid can have different
            # durations even though the timing system resolves a single value.
            _venice_video_cfg = dict(venice_cfg_dict.get('video') or {}) if isinstance(venice_cfg_dict.get('video'), dict) else {}
            _global_dur = venice_duration or resolved_duration

            def _op_duration(op_key: str) -> float:
                op_block = _venice_video_cfg.get(op_key) or {}
                raw = op_block.get('duration_seconds') or op_block.get('duration')
                if raw is None:
                    return _global_dur
                try:
                    return float(str(raw).rstrip('s'))
                except (TypeError, ValueError):
                    return _global_dur

            text2vid_dur = _op_duration('text2vid')
            img2vid_dur  = _op_duration('img2vid')

            # v0.9.1 — when crowd_use_reference_mode is active and a prior
            # lastframe exists, this list is populated with [last_frame_path]
            # and forwarded into vid_req.reference_image_paths so the venice
            # backend attaches it as a `reference_image_urls` payload field
            # (subject to model support).
            crowd_reference_paths: Optional[List[Path]] = None
            requested_inject_source_mode = (
                resolved_cycle_prompt.inject_source_mode if crowd_active else "none"
            )
            actual_inject_source_mode = requested_inject_source_mode if crowd_active else "none"
            reference_images_requested = bool(crowd_active and requested_inject_source_mode == "as_reference")
            reference_images_accepted = False
            reference_images_stripped_by_retry = False
            as_reference_fallback_to_image_source = False
            smart_enabled = smart_reinject_state.enabled
            smart_prefetch = None
            smart_prefetched_keyframe: Optional[Path] = None
            smart_prefetched_valid = False
            smart_similarity_score: Optional[float] = None
            smart_schedule_source_image: Optional[Path] = None
            smart_next_window_prompt: Optional[str] = None
            if smart_enabled and cycle_idx > 1:
                pending = smart_reinject_state.pending
                pending_for_cycle = bool(
                    pending is not None and pending.target_cycle_idx == cycle_idx
                )
                if pending_for_cycle:
                    briq_data["smart_reinject_pending_detected"] = True

                if not smart_reinject_consume_allowed_for_cycle:
                    if pending_for_cycle:
                        if _discard_pending_smart_reinject_for_cycle(
                            state=smart_reinject_state,
                            cycle_idx=cycle_idx,
                            reason="crowd_prompt_active",
                        ):
                            briq_data["smart_reinject_discarded_due_to_crowd"] = True
                            logger.info(
                                "[SlidingStory/SmartReinject] Crowd prompt active for this cycle; "
                                "pending smart prefetch was discarded and will not be consumed."
                            )
                else:
                    wait_timeout = float(getattr(config, "smart_reinject_wait_timeout_sec", 0.0) or 0.0)
                    smart_prefetch = _collect_smart_reinject_prefetch(
                        state=smart_reinject_state,
                        cycle_idx=cycle_idx,
                        keyframes_dir=keyframes_dir,
                        wait_timeout_sec=wait_timeout,
                    )
                    if (
                        pending_for_cycle
                        and smart_prefetch is None
                        and wait_timeout <= 0.0
                        and smart_reinject_state.pending is None
                    ):
                        briq_data["smart_reinject_missed_target_cycle"] = True
                    if smart_prefetch:
                        smart_prefetched_keyframe = Path(smart_prefetch["keyframe_path"])
                        briq_data["smart_reinject_keyframe"] = str(smart_prefetched_keyframe)
                        briq_data["smart_reinject_source_cycle"] = int(smart_prefetch["source_cycle_idx"])

            if cycle_idx == 1:
                if base_image_path and base_image_path.exists() and not crowd_replace_active:
                    # ── image mode: always img2vid from base image ────────
                    venice_mode_str = "image_to_video"
                    image_for_venice = base_image_path
                    briq_data["input_mode"] = "image"
                    logger.info(f"[SlidingStory/Venice] Cycle 1: image_to_video from base image")
                elif venice_prefers_t2v:
                    # ── text mode, prefers t2v: direct text_to_video ──────
                    briq_data["input_mode"] = "text"
                    logger.info(f"[SlidingStory/Venice] Cycle 1: text_to_video")
                else:
                    # ── text mode, prefers img2vid: text2img → img2vid ────
                    # Generate a keyframe from the text prompt, then use it as
                    # the source image for an img2vid call, so cycle 1 is
                    # visually anchored even without a user-supplied base image.
                    logger.info(f"[SlidingStory/Venice] Cycle 1: text2img → img2vid (generating anchor keyframe)")
                    c1kf_req = GenerationRequest(
                        prompt=stacked_prompt,
                        negative_prompt="",
                        seed=seed,
                        mode=InputMode.TEXT,
                        width=get_capability_setting('image', 'width', 1280),
                        height=get_capability_setting('image', 'height', 720),
                        output_dir=keyframes_dir,
                        atom_id=f"{atom_id}_c1kf",
                    )
                    c1kf_result = backend.generate_image(c1kf_req)
                    if c1kf_result.success and c1kf_result.image_path:
                        c1kf_path = keyframes_dir / f"keyframe_c1_anchor.png"
                        if c1kf_result.image_path != c1kf_path:
                            shutil.move(str(c1kf_result.image_path), str(c1kf_path))
                        logger.info(f"[SlidingStory/Venice] Cycle 1: anchor keyframe generated → {c1kf_path}")
                        venice_mode_str = "image_to_video"
                        image_for_venice = c1kf_path
                        briq_data["input_mode"] = "image"
                    else:
                        logger.warning(
                            "[SlidingStory/Venice] Cycle 1: text2img failed, "
                            "falling back to text_to_video"
                        )
                        briq_data["input_mode"] = "text"
                if crowd_active and requested_inject_source_mode == "as_reference":
                    # Cycle 1 has no previous lastframe to attach as reference.
                    actual_inject_source_mode = "text_only_fallback"
            else:
                smart_mode_applied = False
                if (
                    smart_reinject_consume_allowed_for_cycle
                    and smart_enabled
                    and smart_prefetched_keyframe
                    and smart_prefetched_keyframe.exists()
                    and last_frame_path
                    and last_frame_path.exists()
                ):
                    smart_prefetched_valid = True
                    if bool(getattr(config, "smart_reinject_similarity_guard_enabled", True)):
                        try:
                            smart_similarity_score = calculate_frame_similarity(
                                str(last_frame_path),
                                str(smart_prefetched_keyframe),
                            )
                            briq_data["smart_reinject_similarity"] = smart_similarity_score
                            smart_threshold = float(getattr(config, "smart_reinject_similarity_threshold", 0.42))
                            if smart_similarity_score < smart_threshold:
                                smart_prefetched_valid = False
                                briq_data["smart_reinject_rejected_keyframe"] = str(smart_prefetched_keyframe)
                                briq_data["smart_reinject_rejected_reason"] = (
                                    f"similarity<{smart_threshold:.3f}"
                                )
                                logger.info(
                                    "[SlidingStory/SmartReinject] Rejected prefetched keyframe "
                                    f"for cycle {cycle_idx}: similarity={smart_similarity_score:.3f} "
                                    f"< threshold={smart_threshold:.3f}"
                                )
                        except Exception as e:
                            smart_prefetched_valid = False
                            briq_data["smart_reinject_rejected_keyframe"] = str(smart_prefetched_keyframe)
                            briq_data["smart_reinject_rejected_reason"] = f"similarity_error:{e}"
                            logger.warning(
                                f"[SlidingStory/SmartReinject] Similarity guard failed for cycle {cycle_idx}: {e}"
                            )

                    if smart_prefetched_valid:
                        if bool(getattr(config, "smart_reinject_use_morph", True)):
                            if not bool(venice_cfg_dict.get("enable_end_frame_morph", True)):
                                if not smart_reinject_state.warned_end_frame_morph_disabled:
                                    logger.warning(
                                        "[SlidingStory/SmartReinject] smart_reinject_use_morph=true but "
                                        "venice.enable_end_frame_morph=false. Falling back to image_to_video."
                                    )
                                    smart_reinject_state.warned_end_frame_morph_disabled = True
                            else:
                                venice_mode_str = "smart_async_first_last_frame"
                                image_for_venice = last_frame_path
                                end_image_for_venice = smart_prefetched_keyframe
                                briq_data["input_mode"] = "image"
                                briq_data.setdefault("paths", {})
                                briq_data["paths"]["start_image"] = str(last_frame_path)
                                briq_data["paths"]["keyframe"] = str(smart_prefetched_keyframe)
                                briq_data["smart_reinject_used"] = True
                                smart_mode_applied = True
                                smart_schedule_source_image = smart_prefetched_keyframe
                        else:
                            venice_mode_str = "smart_async_image_to_video"
                            image_for_venice = smart_prefetched_keyframe
                            briq_data["input_mode"] = "image"
                            briq_data.setdefault("paths", {})
                            briq_data["paths"]["start_image"] = str(smart_prefetched_keyframe)
                            briq_data["smart_reinject_used"] = True
                            smart_mode_applied = True
                            smart_schedule_source_image = smart_prefetched_keyframe

                if not smart_mode_applied and effective_require_morph and last_frame_path and last_frame_path.exists() and (
                    (not smart_enabled) or bool(getattr(config, "smart_reinject_sync_fallback", False))
                ):
                    rng = random.Random(random_seed_base + cycle_idx * 10007)
                    denoise = rng.uniform(config.img2img_denoise_min, config.img2img_denoise_max)
                    briq_data["denoise"] = denoise

                    logger.info(
                        f"[SlidingStory/Venice] Cycle {cycle_idx}: first/last-frame transition "
                        f"(denoise={denoise:.3f})"
                    )

                    kf_req = GenerationRequest(
                        prompt=stacked_prompt,
                        negative_prompt="",
                        seed=seed,
                        mode=InputMode.IMAGE,
                        init_image_path=last_frame_path,
                        denoise_strength=denoise,
                        width=get_capability_setting('image', 'width', 1280),
                        height=get_capability_setting('image', 'height', 720),
                        output_dir=keyframes_dir,
                        atom_id=f"{atom_id}_kf",
                    )
                    kf_result = backend.generate_image(kf_req)
                    if kf_result.success and kf_result.image_path:
                        keyframe_name = f"keyframe_{cycle_idx:03d}.png"
                        keyframe_path = keyframes_dir / keyframe_name
                        if kf_result.image_path != keyframe_path:
                            shutil.move(str(kf_result.image_path), str(keyframe_path))
                        logger.info(f"[SlidingStory/Venice] Evolved target keyframe → {keyframe_path}")
                        venice_mode_str = "first_last_frame"
                        image_for_venice = last_frame_path
                        end_image_for_venice = keyframe_path
                        briq_data["input_mode"] = "image"
                        briq_data["paths"] = {
                            "keyframe": str(keyframe_path),
                            "start_image": str(last_frame_path),
                        }
                    else:
                        logger.warning("[SlidingStory/Venice] Evolved keyframe failed, falling back to image_to_video")
                        venice_mode_str = "image_to_video"
                        image_for_venice = last_frame_path
                        briq_data["input_mode"] = "image"
                elif not smart_mode_applied and crowd_use_reference_mode:
                    # v0.9.1 — crowd_inject_source_mode == "as_reference":
                    # cycle is text_to_video with the previous lastframe sent
                    # as a REFERENCE image (when the model supports it). The
                    # audience prompt (replace mode) or audience prompt
                    # appended to the story window (append mode) drives the
                    # visual content; the reference acts as a soft style /
                    # identity tether rather than hard frame continuity.
                    supports_refs: Optional[bool] = None
                    if hasattr(backend, "supports_video_optional_field"):
                        try:
                            supports_refs = backend.supports_video_optional_field(
                                op="text2vid",
                                field="reference_image_urls",
                            )
                        except Exception:
                            supports_refs = None
                    if last_frame_path and last_frame_path.exists() and supports_refs is False:
                        venice_mode_str = "image_to_video"
                        image_for_venice = last_frame_path
                        briq_data["input_mode"] = "image"
                        actual_inject_source_mode = "as_image_source"
                        as_reference_fallback_to_image_source = True
                        logger.info(
                            "[SlidingStory/Venice] Cycle "
                            f"{cycle_idx}: crowd as_reference unsupported; falling back to image_to_video "
                            "using previous last frame as source."
                        )
                    else:
                        venice_mode_str = "text_to_video"
                        briq_data["input_mode"] = "text"
                        if last_frame_path and last_frame_path.exists():
                            crowd_reference_paths = [last_frame_path]
                            logger.info(
                                f"[SlidingStory/Venice] Cycle {cycle_idx}: text_to_video "
                                f"(crowd as_reference, lastframe attached as REFERENCE image)"
                            )
                        else:
                            crowd_reference_paths = None
                            actual_inject_source_mode = "text_only_fallback"
                            logger.info(
                                f"[SlidingStory/Venice] Cycle {cycle_idx}: text_to_video "
                                f"(crowd as_reference, no prior frame to use as reference)"
                            )
                elif not smart_mode_applied and last_frame_path and last_frame_path.exists():
                    venice_mode_str = "image_to_video"
                    image_for_venice = last_frame_path
                    briq_data["input_mode"] = "image"
                    if crowd_active and resolved_cycle_prompt.inject_source_mode == "as_image_source":
                        logger.info(
                            f"[SlidingStory/Venice] Cycle {cycle_idx}: image_to_video "
                            f"from last frame (crowd as_image_source)"
                        )
                    else:
                        logger.info(f"[SlidingStory/Venice] Cycle {cycle_idx}: image_to_video from last frame")
                elif not smart_mode_applied:
                    venice_mode_str = "text_to_video"
                    briq_data["input_mode"] = "text"
                    if crowd_active and resolved_cycle_prompt.inject_source_mode == "as_reference":
                        actual_inject_source_mode = "text_only_fallback"
                    logger.info(f"[SlidingStory/Venice] Cycle {cycle_idx}: text_to_video (no prior frame)")

            vid_atom_id = f"video_{cycle_idx:03d}"
            # Select per-op duration: text2vid vs img2vid have independent settings.
            cycle_duration = img2vid_dur if image_for_venice is not None else text2vid_dur
            vid_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.IMAGE if image_for_venice else InputMode.TEXT,
                init_image_path=image_for_venice,
                width=get_capability_setting('video', 'width', 1280),
                height=get_capability_setting('video', 'height', 720),
                output_dir=videos_dir,
                atom_id=vid_atom_id,
                duration_seconds=cycle_duration,
                video_fps=resolved_fps,
                video_frames=resolved_frames,
                video_prompt=video_stage_prompt,
                motion_prompt=motion_prompt_text or None,
                # v0.9.1 — crowd as_reference mode passes lastframe via
                # reference_image_paths; venice backend will translate this
                # into payload["reference_image_urls"] when the model supports it.
                reference_image_paths=crowd_reference_paths,
            )
            if smart_enabled:
                if smart_schedule_source_image is None:
                    if image_for_venice and image_for_venice.exists():
                        smart_schedule_source_image = image_for_venice
                    elif last_frame_path and last_frame_path.exists():
                        smart_schedule_source_image = last_frame_path

                if cycle_idx < total_cycles:
                    next_window_indices = windows[cycle_idx]
                    next_stacked_prompt = "\n\n".join(paragraphs[i - 1] for i in next_window_indices)
                    smart_prefix = str(getattr(config, "smart_reinject_prompt_prefix", "") or "").strip()
                    smart_next_window_prompt = (
                        f"{smart_prefix}\n\n{next_stacked_prompt}" if smart_prefix else next_stacked_prompt
                    )

                    denoise_min = (
                        float(config.smart_reinject_denoise_min)
                        if config.smart_reinject_denoise_min is not None
                        else float(config.img2img_denoise_min)
                    )
                    denoise_max = (
                        float(config.smart_reinject_denoise_max)
                        if config.smart_reinject_denoise_max is not None
                        else float(config.img2img_denoise_max)
                    )
                    if denoise_min > denoise_max:
                        denoise_min, denoise_max = denoise_max, denoise_min
                    smart_rng = random.Random(random_seed_base + cycle_idx * 10007 + 731)
                    smart_denoise = smart_rng.uniform(denoise_min, denoise_max)

                    if smart_reinject_schedule_allowed_for_cycle and _should_schedule_smart_reinject(
                        state=smart_reinject_state,
                        config=config,
                        cycle_idx=cycle_idx,
                        total_cycles=total_cycles,
                        effective_reinject=effective_reinject,
                        crowd_active=crowd_active,
                        source_image_path=smart_schedule_source_image,
                    ):
                        scheduled = _start_smart_reinject_prefetch(
                            state=smart_reinject_state,
                            cycle_idx=cycle_idx,
                            target_cycle_idx=cycle_idx + 1,
                            source_image_path=smart_schedule_source_image,
                            prompt=smart_next_window_prompt,
                            seed=seed + 100000 + cycle_idx,
                            denoise_strength=smart_denoise,
                            width=get_capability_setting('image', 'width', 1280),
                            height=get_capability_setting('image', 'height', 720),
                            output_dir=keyframes_dir,
                        )
                        if scheduled:
                            briq_data["smart_reinject_scheduled"] = True
                            briq_data["smart_reinject_scheduled_for_cycle"] = cycle_idx + 1
                            briq_data["smart_reinject_schedule_source"] = str(smart_schedule_source_image)
                            briq_data["smart_reinject_prefetch_denoise"] = smart_denoise
                            briq_data["smart_reinject_prompt_preview"] = smart_next_window_prompt[:220]
                    else:
                        briq_data["smart_reinject_scheduled"] = False
                        if crowd_active and smart_enabled:
                            logger.info(
                                "[SlidingStory/SmartReinject] Smart reinject paused for this cycle "
                                "because crowd prompt is active; scheduling skipped."
                            )

            if venice_mode_str in {"first_last_frame", "smart_async_first_last_frame"} and image_for_venice and end_image_for_venice:
                vid_result = backend.generate_morph_video(
                    vid_req,
                    start_image_path=image_for_venice,
                    end_image_path=end_image_for_venice,
                )
                if (
                    venice_mode_str == "smart_async_first_last_frame"
                    and not (vid_result and vid_result.success and vid_result.video_path and vid_result.video_path.exists())
                    and image_for_venice
                ):
                    fallback_error = getattr(vid_result, "error", "unknown error")
                    if not smart_reinject_state.warned_morph_capability_failure:
                        logger.warning(
                            "[SlidingStory/SmartReinject] Morph call failed for prefetched keyframe; "
                            "falling back to image_to_video from last frame."
                        )
                        smart_reinject_state.warned_morph_capability_failure = True
                    logger.warning(
                        f"[SlidingStory/SmartReinject] Morph fallback reason (cycle {cycle_idx}): {fallback_error}"
                    )
                    vid_result = backend.generate_video(
                        vid_req,
                        source_image=image_for_venice,
                    )
                    venice_mode_str = "smart_async_first_last_frame_fallback_i2v"
                    briq_data["smart_reinject_used"] = False
                    briq_data["smart_reinject_fallback_reason"] = str(fallback_error)
            else:
                vid_result = backend.generate_video(
                    vid_req,
                    source_image=image_for_venice,
                )

            venice_response_meta = ((getattr(vid_result, "metadata", None) or {}).get("response") or {})
            stripped_retry_fields = set(venice_response_meta.get("stripped_retry_fields") or [])
            accepted_optional_fields = set(venice_response_meta.get("accepted_optional_fields") or [])
            reference_images_stripped_by_retry = "reference_image_urls" in stripped_retry_fields
            reference_images_accepted = (
                "reference_image_urls" in accepted_optional_fields and not reference_images_stripped_by_retry
            )

            if (
                crowd_active
                and requested_inject_source_mode == "as_reference"
                and actual_inject_source_mode == "as_reference"
                and reference_images_stripped_by_retry
                and last_frame_path
                and last_frame_path.exists()
            ):
                logger.warning(
                    "[SlidingStory/Venice] reference_image_urls was stripped by Venice retry; "
                    "falling back to image_to_video for this cycle."
                )
                fallback_req = GenerationRequest(
                    prompt=stacked_prompt,
                    negative_prompt="",
                    seed=seed,
                    mode=InputMode.IMAGE,
                    init_image_path=last_frame_path,
                    width=get_capability_setting('video', 'width', 1280),
                    height=get_capability_setting('video', 'height', 720),
                    output_dir=videos_dir,
                    atom_id=vid_atom_id,
                    duration_seconds=img2vid_dur,
                    video_fps=resolved_fps,
                    video_frames=resolved_frames,
                    video_prompt=video_stage_prompt,
                    motion_prompt=motion_prompt_text or None,
                    reference_image_paths=None,
                )
                fallback_result = backend.generate_video(
                    fallback_req,
                    source_image=last_frame_path,
                )
                if fallback_result and fallback_result.success and fallback_result.video_path and fallback_result.video_path.exists():
                    vid_result = fallback_result
                    venice_mode_str = "image_to_video_reference_fallback"
                    actual_inject_source_mode = "as_image_source"
                    as_reference_fallback_to_image_source = True
                    reference_images_accepted = False
                else:
                    actual_inject_source_mode = "text_only_fallback"
                    logger.warning(
                        "[SlidingStory/Venice] as_reference fallback to image_to_video failed; "
                        "keeping text-only result for this cycle."
                    )

            briq_data["requested_inject_source_mode"] = requested_inject_source_mode
            briq_data["actual_inject_source_mode"] = actual_inject_source_mode
            briq_data["reference_images_requested"] = reference_images_requested
            briq_data["reference_images_accepted"] = bool(reference_images_accepted)
            briq_data["reference_images_stripped_by_retry"] = bool(reference_images_stripped_by_retry)
            if as_reference_fallback_to_image_source:
                briq_data["as_reference_fallback_to_image_source"] = True

            if not (vid_result and vid_result.success and vid_result.video_path and vid_result.video_path.exists()):
                error_msg = getattr(vid_result, 'error', 'unknown error')
                raise RuntimeError(f"[Venice] Video generation failed in cycle {cycle_idx}: {error_msg}")

            video_name = f"video_{cycle_idx:03d}.mp4"
            video_path = videos_dir / video_name
            if vid_result.video_path != video_path:
                shutil.move(str(vid_result.video_path), str(video_path))
            logger.info(f"[SlidingStory/Venice] Generated video → {video_path}")

            current_cycle_video_path = video_path
            if per_cycle_processing_enabled:
                interpolated_video_path = finalizer._process_cycle_video(
                    video_path, resolved_fps
                )
                if interpolated_video_path:
                    current_cycle_video_path = interpolated_video_path

            # When pingpong is on, extract last frame from the original
            # (pre-pingpong) video — its actual last frame is the true "end
            # of new visual content". The pingpong'd video's last frame
            # equals its first frame, which would loop the story.
            lastframe_source = (
                video_path if per_cycle_pingpong_enabled else current_cycle_video_path
            )
            lastframe_name = f"lastframe_{cycle_idx:03d}.png"
            lastframe_path = keyframes_dir / lastframe_name
            _extract_last_frame_ffmpeg(lastframe_source, lastframe_path)

            # ── Post-resize extracted frame to configured target dims ──────────
            # Seedance (and some other img2vid models) output a fixed native
            # resolution regardless of input image size. Resize extracted frames
            # to the configured img2vid target so all subsequent cycles receive
            # the correct source dimensions.
            _venice_frame_target = None
            try:
                from .venice_backend import VeniceBackend as _VB
                _vc_dict = config.venice_config or backend_cfg.get('venice', {})
                _vc_video = dict(_vc_dict.get('video') or {}) if isinstance(_vc_dict.get('video'), dict) else {}
                _op_block = dict(_vc_video.get('img2vid') or {})
                _res_tok = _op_block.get('resolution') or _vc_video.get('resolution') or '720p'
                _ar_str = str(_op_block.get('aspect_ratio') or _vc_video.get('aspect_ratio') or '16:9')
                _venice_frame_target = _VB._resolution_to_dims(_res_tok, _ar_str)
            except Exception:
                pass
            if _venice_frame_target:
                _resize_frame_to_target(lastframe_path, *_venice_frame_target)

            last_frame_path = lastframe_path
            final_video_paths.append(current_cycle_video_path)

            if cycle_idx == 1:
                anchor_frame_path = keyframes_dir / "anchor_frame_001.png"
                _extract_first_frame_ffmpeg(current_cycle_video_path, anchor_frame_path)
                if _venice_frame_target:
                    _resize_frame_to_target(anchor_frame_path, *_venice_frame_target)
                logger.debug(f"[SlidingStory] Anchor frame saved → {anchor_frame_path.name}")

            briq_data.setdefault("paths", {})
            briq_data["paths"]["video"] = str(current_cycle_video_path)
            briq_data["paths"]["last_frame"] = str(lastframe_path)
            briq_data["venice_mode"] = venice_mode_str
            if vid_result.metadata:
                briq_data["venice_metadata"] = vid_result.metadata
            _write_briq_json(briqs_dir, cycle_idx, briq_data)
            if checkpoint_callback:
                checkpoint_callback(cycle_idx, last_frame_path, current_cycle_video_path, anchor_frame_path)
            continue  # Skip ComfyUI cycle logic below

        # ═══════════════════════════════════════════════════════════════════
        # COMFYUI / MOCK BACKEND CYCLE PATH (original v0.5.x logic)
        # ═══════════════════════════════════════════════════════════════════
        # === Step 1: Keyframe generation ===
        if cycle_idx == 1:
            if base_image_path and base_image_path.exists() and not crowd_replace_active:
                # Image/Video mode: use base image as start for cycle 0.
                # Skipped when crowd_replace_active — must do a full visual reset.
                logger.info(f"[SlidingStory] Using base image for cycle 1: {base_image_path}")
                briq_data["input_mode"] = "image"

                # Run img2img on base image with story prompt
                rng = random.Random(random_seed_base + cycle_idx * 10007)
                denoise = rng.uniform(config.img2img_denoise_min, config.img2img_denoise_max)
                briq_data["denoise"] = denoise

                img_req = GenerationRequest(
                    prompt=stacked_prompt,
                    negative_prompt="",
                    seed=seed,
                    mode=InputMode.IMAGE,
                    init_image_path=base_image_path,
                    denoise_strength=denoise,
                    width=get_capability_setting('image', 'width', 1280),
                    height=get_capability_setting('image', 'height', 720),
                    output_dir=comfy_tmp,
                    atom_id=atom_id,
                )
                result: GenerationResult = backend.generate_image(img_req)
                if not result.success or not result.image_path:
                    raise RuntimeError(f"Keyframe generation failed in cycle {cycle_idx}: {result.error}")
                keyframe_name = f"keyframe_{cycle_idx:03d}.png"
                keyframe_path = keyframes_dir / keyframe_name
                shutil.copyfile(result.image_path, keyframe_path)
                briq_data["paths"] = {"keyframe": str(keyframe_path)}
            else:
                # Text mode: Text2Img for the first keyframe.
                # Also the path taken when crowd_replace_active=True (even if base_image exists).
                if crowd_replace_active:
                    logger.info(
                        "[SlidingStory] Cycle 1: crowd REPLACE active — "
                        "forcing TEXT2IMG (ignoring base image)"
                    )
                briq_data["input_mode"] = "text"
                req = GenerationRequest(
                    prompt=stacked_prompt,
                    negative_prompt="",
                    seed=seed,
                    mode=InputMode.TEXT,
                    width=get_capability_setting('image', 'width', 1280),
                    height=get_capability_setting('image', 'height', 720),
                    output_dir=comfy_tmp,
                    atom_id=atom_id,
                )
                result: GenerationResult = backend.generate_image(req)
                if not result.success or not result.image_path:
                    raise RuntimeError(f"Keyframe generation failed in cycle {cycle_idx}: {result.error}")
                # Copy keyframe
                keyframe_name = f"keyframe_{cycle_idx:03d}.png"
                keyframe_path = keyframes_dir / keyframe_name
                shutil.copyfile(result.image_path, keyframe_path)
                briq_data["paths"] = {"keyframe": str(keyframe_path)}
            logger.info(f"[SlidingStory] Generated keyframe → {keyframe_path}")
            # Generate initial motion video from keyframe
            vid_atom_id = f"video_{cycle_idx:03d}"
            vid_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.IMAGE,
                width=get_capability_setting('video', 'width', 1280),
                height=get_capability_setting('video', 'height', 720),
                output_dir=comfy_tmp,
                atom_id=vid_atom_id,
                duration_seconds=resolved_duration,
                video_fps=resolved_fps,
                video_frames=resolved_frames,
                video_prompt=video_stage_prompt,
                motion_prompt=motion_prompt_text or None,
            )
            gen_vid: GenerationResult = _render_video_with_continuity_guard(
                backend=backend,
                request=vid_req,
                config=config,
                require_morph=False,
                conditioning_image=keyframe_path,
            )
            video_name = f"video_{cycle_idx:03d}.mp4"
            video_path = videos_dir / video_name
            if not (gen_vid.success and gen_vid.video_path and gen_vid.video_path.exists()):
                raise RuntimeError(
                    f"Initial video generation failed in cycle {cycle_idx}: {gen_vid.error}"
                )
            shutil.copyfile(gen_vid.video_path, video_path)
            logger.info(f"[SlidingStory] Generated initial video → {video_path}")

            current_cycle_video_path = video_path # Default to raw backend video

            if per_cycle_processing_enabled:
                interpolated_video_path = finalizer._process_cycle_video(
                    video_path, resolved_fps
                )
                if interpolated_video_path:
                    current_cycle_video_path = interpolated_video_path
                    logger.info(f"[SlidingStory] Using processed video → {current_cycle_video_path}")
                else:
                    logger.warning(
                        f"[SlidingStory] Per-cycle processing failed for {video_path.name}. "
                        "Falling back to raw backend video."
                    )
            
            # Extract last frame. When pingpong is on, take it from the
            # original (pre-pingpong) video so subsequent cycles don't loop
            # on themselves (pingpong's last frame == its first frame).
            lastframe_source = (
                video_path if per_cycle_pingpong_enabled else current_cycle_video_path
            )
            lastframe_name = f"lastframe_{cycle_idx:03d}.png"
            lastframe_path = keyframes_dir / lastframe_name
            _extract_last_frame_ffmpeg(lastframe_source, lastframe_path)
            last_frame_path = lastframe_path
            if cycle_idx == 1 and not anchor_frame_path and keyframe_path.exists():
                anchor_frame_path = keyframes_dir / "anchor_frame_001.png"
                shutil.copyfile(keyframe_path, anchor_frame_path)
                logger.debug(f"[SlidingStory] Anchor frame saved → {anchor_frame_path.name}")
            final_video_paths.append(current_cycle_video_path)
            briq_data["paths"]["video"] = str(current_cycle_video_path)
            briq_data["paths"]["last_frame"] = str(lastframe_path)
            _write_briq_json(briqs_dir, cycle_idx, briq_data)
            # ── Checkpoint callback (v0.6.1-beta) ────────────────────────
            if checkpoint_callback:
                checkpoint_callback(cycle_idx, last_frame_path, current_cycle_video_path, anchor_frame_path)
            continue  # Move to next cycle

        # === Subsequent cycles ===
        rng = random.Random(random_seed_base + cycle_idx * 10007)
        denoise = rng.uniform(config.img2img_denoise_min, config.img2img_denoise_max)
        briq_data["denoise"] = denoise

        if crowd_replace_active:
            # CROWD REPLACE: full visual reset — TEXT2IMG, no init image, no last-frame bleed
            logger.info(
                "[SlidingStory] Crowd REPLACE visual reset: TEXT2IMG keyframe "
                "(reinject overridden)"
            )
            img_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.TEXT,
                init_image_path=None,
                width=get_capability_setting('image', 'width', 1280),
                height=get_capability_setting('image', 'height', 720),
                output_dir=comfy_tmp,
                atom_id=atom_id,
            )
            img_result = backend.generate_image(img_req)
            if not img_result.success or not img_result.image_path:
                raise RuntimeError(f"TEXT2IMG generation failed in cycle {cycle_idx}: {img_result.error}")
            keyframe_name = f"keyframe_{cycle_idx:03d}.png"
            keyframe_path = keyframes_dir / keyframe_name
            shutil.copyfile(img_result.image_path, keyframe_path)
            logger.info(f"[SlidingStory] Generated keyframe (crowd reset) → {keyframe_path}")
            # No start_image reference — this cycle started fresh
            briq_data["paths"] = {"keyframe": str(keyframe_path)}
            conditioning_image = keyframe_path
        elif effective_reinject:
            # REINJECT MODE (default): img2img keyframe from last frame
            logger.info(f"[SlidingStory] Reinject: img2img keyframe (denoise={denoise:.3f})")
            img_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.IMAGE,
                init_image_path=last_frame_path,
                denoise_strength=denoise,
                width=get_capability_setting('image', 'width', 1280),
                height=get_capability_setting('image', 'height', 720),
                output_dir=comfy_tmp,
                atom_id=atom_id,
            )
            img_result = backend.generate_image(img_req)
            if not img_result.success or not img_result.image_path:
                raise RuntimeError(f"Img2Img generation failed in cycle {cycle_idx}: {img_result.error}")
            keyframe_name = f"keyframe_{cycle_idx:03d}.png"
            keyframe_path = keyframes_dir / keyframe_name
            shutil.copyfile(img_result.image_path, keyframe_path)
            logger.info(f"[SlidingStory] Generated keyframe → {keyframe_path}")
            briq_data["paths"] = {"keyframe": str(keyframe_path), "start_image": str(last_frame_path)}
            conditioning_image = keyframe_path
        else:
            # NO-REINJECT: use last frame directly as conditioning
            logger.info("[SlidingStory] No-reinject: using last frame directly")
            conditioning_image = last_frame_path
            briq_data["paths"] = {"start_image": str(last_frame_path)}

        # === Generate transition video ===
        video_name = f"video_{cycle_idx:03d}.mp4"
        video_path = videos_dir / video_name
        vid_atom_id = f"video_{cycle_idx:03d}"
        vid_req = GenerationRequest(
            prompt=stacked_prompt,
            negative_prompt="",
            seed=seed,
            mode=InputMode.IMAGE,
            width=get_capability_setting('morph' if effective_require_morph else 'video', 'width', 1280),
            height=get_capability_setting('morph' if effective_require_morph else 'video', 'height', 720),
            output_dir=comfy_tmp,
            atom_id=vid_atom_id,
            duration_seconds=resolved_duration,
            video_fps=resolved_fps,
            video_frames=resolved_frames,
        )

        vid_result: Optional[GenerationResult] = None
        if effective_require_morph:
            # morph requires both chain endpoints; crowd replace resets so it doesn't run
            logger.info("[SlidingStory] Generating morph video (require_morph=true)")
            try:
                vid_result = _render_video_with_continuity_guard(
                    backend=backend,
                    request=vid_req,
                    config=config,
                    require_morph=True,
                    conditioning_image=conditioning_image,
                    last_frame_path=last_frame_path,
                )
            except Exception as e:
                raise RuntimeError(f"Morph video generation failed in cycle {cycle_idx}: {e}")
        else:
            logger.info("[SlidingStory] Generating standard video")
            vid_result = _render_video_with_continuity_guard(
                backend=backend,
                request=vid_req,
                config=config,
                require_morph=False,
                conditioning_image=conditioning_image,
            )

        if not (vid_result and vid_result.success and vid_result.video_path and vid_result.video_path.exists()):
            error_msg = getattr(vid_result, 'error', 'unknown error')
            raise RuntimeError(f"Video generation failed in cycle {cycle_idx}: {error_msg}")

        shutil.copyfile(vid_result.video_path, video_path)
        logger.info(f"[SlidingStory] Generated video → {video_path}")

        current_cycle_video_path = video_path # Default to raw backend video

        if per_cycle_processing_enabled:
            interpolated_video_path = finalizer._process_cycle_video(
                video_path, resolved_fps
            )
            if interpolated_video_path:
                current_cycle_video_path = interpolated_video_path
                logger.info(f"[SlidingStory] Using processed video → {current_cycle_video_path}")
            else:
                logger.warning(
                    f"[SlidingStory] Per-cycle processing failed for {video_path.name}. "
                    "Falling back to raw backend video."
                )

        # Extract last frame. When pingpong is on, take it from the original
        # (pre-pingpong) video so subsequent cycles don't loop on themselves.
        lastframe_source = (
            video_path if per_cycle_pingpong_enabled else current_cycle_video_path
        )
        lastframe_name = f"lastframe_{cycle_idx:03d}.png"
        lastframe_path = keyframes_dir / lastframe_name
        _extract_last_frame_ffmpeg(lastframe_source, lastframe_path)
        last_frame_path = lastframe_path
        final_video_paths.append(current_cycle_video_path)
        briq_data["paths"]["video"] = str(current_cycle_video_path)
        briq_data["paths"]["last_frame"] = str(lastframe_path)
        _write_briq_json(briqs_dir, cycle_idx, briq_data)
        # ── Checkpoint callback (v0.6.1-beta) ────────────────────────────
        if checkpoint_callback:
            checkpoint_callback(cycle_idx, last_frame_path, current_cycle_video_path, anchor_frame_path)

    _shutdown_smart_reinject_state(smart_reinject_state)

    # ═══════════════════════════════════════════════════════════════════════
    # LOOP CLOSURE (v0.6.0-beta: Veo, v0.6.7-beta: +LTX-Video)
    # ═══════════════════════════════════════════════════════════════════════
    # If enabled, generate a final clip that transitions from the last frame
    # back to the cycle-1 anchor frame, creating a seamless loop.
    # loop_closure_strength modulates duration: higher → longer clip → smoother.
    if (
        config.enable_loop_closure
        and anchor_frame_path
        and anchor_frame_path.exists()
        and last_frame_path
        and last_frame_path.exists()
    ):
        loop_cycle_idx = total_cycles + 1
        loop_backend_label = (
            "Veo" if morph_is_veo else
            "LTX" if morph_is_ltx else
            "Venice" if morph_is_venice else
            morph_backend_type
        )

        if morph_is_veo:
            veo_orch_lc = config.veo_config or backend_cfg.get('veo', {})
            lc_strength = float(veo_orch_lc.get('loop_closure_strength', 0.90))
        elif morph_is_ltx:
            ltx_lc_cfg = config.ltx_video_config or backend_cfg.get('ltx_video', {})
            lc_strength = float(ltx_lc_cfg.get('loop_closure_strength', 0.90))
        elif morph_is_venice:
            venice_lc_cfg = config.venice_config or backend_cfg.get('venice', {})
            lc_strength = float(venice_lc_cfg.get('loop_closure_strength', 0.90))
        else:
            morph_cfg = (backend_cfg.get('morph_backend') or {}) if isinstance(backend_cfg, dict) else {}
            ps_loop_strength = getattr(config, 'loop_closure_strength', None)
            lc_strength = float(
                morph_cfg.get('loop_closure_strength', backend_cfg.get('loop_closure_strength', ps_loop_strength or 0.90))
            )

        # Map strength to duration: low → 4s, medium → 6s, high → 8s
        lc_duration = 4 if lc_strength < 0.5 else (6 if lc_strength < 0.8 else 8)

        logger.info(
            f"[SlidingStory/{loop_backend_label}] Generating loop closure clip "
            f"(last frame → anchor frame, strength={lc_strength:.2f}, duration={lc_duration}s)"
        )
        loop_atom_id = f"video_loop_{loop_cycle_idx:03d}"
        loop_prompt = "Smooth seamless visual transition, continuous flow"

        loop_req = GenerationRequest(
            prompt=loop_prompt,
            negative_prompt="",
            seed=random_seed_base + loop_cycle_idx,
            mode=InputMode.IMAGE,
            init_image_path=last_frame_path,
            width=get_capability_setting('morph', 'width', 1280),
            height=get_capability_setting('morph', 'height', 720),
            output_dir=videos_dir,
            atom_id=loop_atom_id,
            duration_seconds=lc_duration,
            video_fps=resolved_fps,
            veo_mode="first_last_frame" if morph_is_veo else None,
            last_frame_path=anchor_frame_path if morph_is_veo else None,
        )

        try:
            loop_result = backend.generate_morph_video(
                loop_req,
                start_image_path=last_frame_path,
                end_image_path=anchor_frame_path,
            )
            if loop_result and loop_result.success and loop_result.video_path:
                loop_video_name = f"video_loop_{loop_cycle_idx:03d}.mp4"
                loop_video_path = videos_dir / loop_video_name
                if loop_result.video_path != loop_video_path:
                    shutil.move(str(loop_result.video_path), str(loop_video_path))
                final_video_paths.append(loop_video_path)
                logger.info(f"[SlidingStory/{loop_backend_label}] Loop closure → {loop_video_path}")

                # Write briq for loop closure
                _write_briq_json(briqs_dir, loop_cycle_idx, {
                    "cycle_index": loop_cycle_idx,
                    "type": "loop_closure",
                    "backend": morph_backend_type,
                    "mode": (
                        "conditioned_transition" if morph_is_ltx else
                        "first_last_frame" if morph_is_veo else
                        "morph"
                    ),
                    "paths": {
                        "video": str(loop_video_path),
                        "start_image": str(last_frame_path),
                        "end_image": str(anchor_frame_path),
                    },
                })
            else:
                logger.warning(
                    f"[SlidingStory/{loop_backend_label}] Loop closure failed: "
                    f"{getattr(loop_result, 'error', 'unknown')} — skipping"
                )
        except Exception as e:
            logger.warning(f"[SlidingStory/{loop_backend_label}] Loop closure error (non-fatal): {e}")

    # === Final assembly ===
    if not final_video_paths:
        raise RuntimeError("No video segments were generated; cannot assemble final video")
    
    # Finalize the stitched video (this creates final_output.mp4)
    stitched_master_path = finalizer.finalize(cycle_video_paths=final_video_paths)
    
    # Run post-stitch finalizer (creates final_60fps_1080p.mp4 if enabled)
    final_deliverable_path = finalizer.run_post_stitch_finalizer()
    
    logger.info(f"[SlidingStory] Final story video assembled → {stitched_master_path}")
    if final_deliverable_path:
        logger.info(f"[SlidingStory] Post-stitch deliverable created → {final_deliverable_path}")
        result_path = final_deliverable_path # Return the highest quality deliverable
    else:
        result_path = stitched_master_path

    # Clean up temporary backend outputs
    try:
        shutil.rmtree(comfy_tmp, ignore_errors=True)
    except Exception:
        pass
    return result_path
