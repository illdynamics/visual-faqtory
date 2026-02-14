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
    implemented on the Visual FaQtory side. ComfyUI (or the chosen
    backend) is never allowed to own directory structure or story state.

  • Only three backend operations are used: text2img (generate_image with
    no init), img2img (generate_image with init_image_path) and
    img2vid (generate_video or internal cross‑fade). All backend
    outputs are written into a temporary directory and then copied into
    qodeyard/ by this module.

  • The runtime directories are flat: qodeyard/story.txt, qodeyard/keyframes,
    qodeyard/lastframes, qodeyard/videos. No subdirectories per run. Any
    previous contents of those directories should be cleared by the caller
    before invoking this engine.

  • The windowing schedule has three phases for P total paragraphs and
    maximum window size M (max_paragraphs):

        – Ramp‑up: cycles 1..M grow the window [1], [1,2], [1,2,3], …
        – Sliding: cycles M+1..P slide a fixed size window of M paragraphs
        – Ramp‑down: after the last paragraph triggers, continue cycles
          dropping the earliest paragraph until only one remains.

    Generation stops after the single‑paragraph window completes.

  • Denoise values for img2img are drawn uniformly from a configurable
    range [img2img_denoise_min, img2img_denoise_max] per cycle.

  • The video transition from one keyframe to the next is implemented
    using a simple cross‑fade between the last frame of the previous
    cycle and the new keyframe. This avoids camera motion and complex
    latent interpolation. The frame rate is derived from the base
    backend configuration or defaults to 8fps if unspecified.

To run the engine, construct a SlidingStoryConfig (see below) and call
run_sliding_story(). For convenience, a CLI entry point is provided
via vfaq_cli (see vfaq_cli.py).

Part of QonQrete Visual FaQtory v0.5.6-beta
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import subprocess
import tempfile

from .backends import (
    create_backend,
    GenerationRequest,
    GenerationResult,
    InputMode,
)

logger = logging.getLogger(__name__)


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
        img2vid_duration_sec: Duration in seconds of each transition video.
        img2img_denoise_min: Minimum denoise_strength for img2img calls.
        img2img_denoise_max: Maximum denoise_strength for img2img calls.
        rolling_window_mode: Enable the sliding window engine when True.
        require_morph:       When true, cycles ≥ 2 must use morphing.
        seed_base:           Base seed used for deterministic generation.
        video_fps:           Frames per second for cross‑fade videos.
        backend_config:      Backend configuration dictionary used to
                             instantiate the generator backend.
    """
    max_paragraphs: int = 4
    img2vid_duration_sec: float = 3.0
    img2img_denoise_min: float = 0.25
    img2img_denoise_max: float = 0.45
    rolling_window_mode: bool = True
    require_morph: bool = False
    seed_base: int = 42
    video_fps: int = 8
    backend_config: Dict[str, any] | None = None
    reinject: bool = True  # When True (default), every cycle runs img2img keyframe


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


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get duration for {video_path}: {e}")
        return 0.0


def _extract_last_frame_ffmpeg(video_path: Path, output_path: Path) -> None:
    """Extract the last frame of a video using ffmpeg.

    Args:
        video_path: Path to the input MP4 video.
        output_path: Path to write the extracted PNG image.

    Raises:
        RuntimeError: If ffmpeg fails to extract the frame.
    """
    if not video_path.exists():
        logger.error(f"[SlidingStory] Cannot extract last frame, video file does not exist: {video_path}")
        raise FileNotFoundError(f"Input video for frame extraction not found: {video_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    if duration == 0.0:
        raise RuntimeError(f"Could not determine duration of video: {video_path}")

    seek_time = max(0, duration - 0.5) # seek to 0.5s before end to be safe

    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(seek_time), # Seek from beginning to (duration - 0.5s)
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',
        str(output_path)
    ]
    logger.info(f"[SlidingStory] Running ffmpeg to extract last frame: {' '.join(cmd)}")
    try:
        # Using capture_output to get stderr in case of non-zero exit code
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
             logger.error(f"[SlidingStory] ffmpeg ran but output file is missing or empty: {output_path}")
             logger.error(f"[SlidingStory] ffmpeg stderr: {result.stderr}")
             raise RuntimeError(f"ffmpeg failed to create a valid output file: {output_path}")
        logger.info(f"[SlidingStory] Successfully extracted last frame to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[SlidingStory] ffmpeg failed with exit code {e.returncode}")
        logger.error(f"[SlidingStory] ffmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Failed to extract last frame via ffmpeg: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during last frame extraction: {e}")
        raise


def _concat_videos_ffmpeg(video_paths: List[Path], output_path: Path) -> None:
    """Concatenate multiple videos into one using ffmpeg concat demuxer.

    Args:
        video_paths: Ordered list of video segment paths to concatenate.
        output_path: Path to the resulting concatenated MP4 file.

    Raises:
        RuntimeError: If ffmpeg fails to concatenate videos.
    """
    # Write a temporary file listing all input videos
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as list_file:
        for vp in video_paths:
            list_file.write(f"file '{vp.as_posix()}'\n")
        list_filename = list_file.name
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_filename,
        '-c', 'copy',
        str(output_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to concatenate videos via ffmpeg: {e}")
    finally:
        # Clean up the temporary list file
        try:
            os.unlink(list_filename)
        except Exception:
            pass


def run_sliding_story(
    story_path: Path,
    qodeyard_dir: Path,
    config: SlidingStoryConfig,
    max_cycles: Optional[int] = None,
    base_image_path: Optional[Path] = None,
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

    Returns:
        Path to the final concatenated story video.
    """
    logger.info(f"[SlidingStory] Starting story run: {story_path}")
    logger.info(f"[SlidingStory] Reinject: {config.reinject}")
    # Ensure output directory structure exists
    qodeyard_dir = qodeyard_dir.resolve()
    keyframes_dir = qodeyard_dir / "frames"
    lastframes_dir = qodeyard_dir / "frames"
    videos_dir = qodeyard_dir / "videos"
    briqs_dir = qodeyard_dir / "briqs"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    briqs_dir.mkdir(parents=True, exist_ok=True)

    # Copy story file into qodeyard
    dest_story = qodeyard_dir / "story.txt"
    shutil.copyfile(story_path, dest_story)
    logger.info(f"[SlidingStory] Story copied to {dest_story}")

    # Parse paragraphs
    paragraphs = _parse_story_file(story_path)
    if not paragraphs:
        raise RuntimeError("Story contains no paragraphs; nothing to generate")
    P = len(paragraphs)
    M = config.max_paragraphs
    windows = _determine_windows(P, M)

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

    # Use a temporary directory for backend outputs.  Do not place this
    # directory inside qodeyard to prevent ComfyUI from writing into
    # user‑controlled spaces. Instead, allocate a temporary directory
    # under the system temp path.
    comfy_tmp = Path(tempfile.mkdtemp(prefix="vfaq_comfyui_tmp_"))

    # Track last frame path for chaining
    last_frame_path: Optional[Path] = None
    final_video_paths: List[Path] = []

    # Seed management for determinism: derive cycle‑specific seeds
    random_seed_base = config.seed_base

    for cycle_idx, window_indices in enumerate(windows, start=1):
        logger.info(f"[SlidingStory] Cycle {cycle_idx}/{total_cycles} — window paragraphs {window_indices}")
        # Concatenate paragraphs as stacked prompt; join with two newlines for clarity
        stacked_prompt = "\n\n".join(paragraphs[i - 1] for i in window_indices)
        logger.debug(f"[SlidingStory] Stacked prompt:\n{stacked_prompt}")

        # Determine unique atom_id for this cycle
        atom_id = f"cycle_{cycle_idx:03d}"

        # Determine seed for this cycle (offset by cycle index)
        seed = random_seed_base + cycle_idx

        # Briq data for this cycle
        briq_data = {
            "cycle_index": cycle_idx,
            "paragraph_window": {"start": window_indices[0], "end": window_indices[-1]},
            "paragraph_text": stacked_prompt[:500],
            "seed": seed,
            "input_mode": "text",
            "reinject": config.reinject,
            "backend_type": backend_cfg.get("type", "mock"),
        }

        # === Step 1: Keyframe generation ===
        if cycle_idx == 1:
            if base_image_path and base_image_path.exists():
                # Image/Video mode: use base image as start for cycle 0
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
                    width=backend_cfg.get('width', 1024),
                    height=backend_cfg.get('height', 576),
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
                # Text mode: Text2Img for the first keyframe
                briq_data["input_mode"] = "text"
                req = GenerationRequest(
                    prompt=stacked_prompt,
                    negative_prompt="",
                    seed=seed,
                    mode=InputMode.TEXT,
                    width=backend_cfg.get('width', 1024),
                    height=backend_cfg.get('height', 576),
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
                width=backend_cfg.get('width', 1024),
                height=backend_cfg.get('height', 576),
                output_dir=comfy_tmp,
                atom_id=vid_atom_id,
                duration_seconds=config.img2vid_duration_sec,
                video_fps=config.video_fps,
            )
            gen_vid: GenerationResult = backend.generate_video(vid_req, keyframe_path)
            video_name = f"video_{cycle_idx:03d}.mp4"
            video_path = videos_dir / video_name
            if not (gen_vid.success and gen_vid.video_path and gen_vid.video_path.exists()):
                raise RuntimeError(
                    f"Initial video generation failed in cycle {cycle_idx}: {gen_vid.error}"
                )
            shutil.copyfile(gen_vid.video_path, video_path)
            logger.info(f"[SlidingStory] Generated initial video → {video_path}")
            # Extract last frame
            lastframe_name = f"lastframe_{cycle_idx:03d}.png"
            lastframe_path = keyframes_dir / lastframe_name
            _extract_last_frame_ffmpeg(video_path, lastframe_path)
            last_frame_path = lastframe_path
            final_video_paths.append(video_path)
            briq_data["paths"]["video"] = str(video_path)
            briq_data["paths"]["last_frame"] = str(lastframe_path)
            _write_briq_json(briqs_dir, cycle_idx, briq_data)
            continue  # Move to next cycle

        # === Subsequent cycles ===
        rng = random.Random(random_seed_base + cycle_idx * 10007)
        denoise = rng.uniform(config.img2img_denoise_min, config.img2img_denoise_max)
        briq_data["denoise"] = denoise

        if config.reinject:
            # REINJECT MODE (default): img2img keyframe from last frame
            logger.info(f"[SlidingStory] Reinject: img2img keyframe (denoise={denoise:.3f})")
            img_req = GenerationRequest(
                prompt=stacked_prompt,
                negative_prompt="",
                seed=seed,
                mode=InputMode.IMAGE,
                init_image_path=last_frame_path,
                denoise_strength=denoise,
                width=backend_cfg.get('width', 1024),
                height=backend_cfg.get('height', 576),
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
            width=backend_cfg.get('width', 1024),
            height=backend_cfg.get('height', 576),
            output_dir=comfy_tmp,
            atom_id=vid_atom_id,
            duration_seconds=config.img2vid_duration_sec,
            video_fps=config.video_fps,
        )

        vid_result: Optional[GenerationResult] = None
        if config.require_morph and config.reinject:
            logger.info("[SlidingStory] Generating morph video (require_morph=true)")
            try:
                vid_result = backend.generate_morph_video(
                    vid_req,
                    start_image_path=last_frame_path,
                    end_image_path=conditioning_image
                )
            except Exception as e:
                raise RuntimeError(f"Morph video generation failed in cycle {cycle_idx}: {e}")
        else:
            logger.info("[SlidingStory] Generating standard video")
            vid_result = backend.generate_video(vid_req, source_image=conditioning_image)

        if not (vid_result and vid_result.success and vid_result.video_path and vid_result.video_path.exists()):
            error_msg = getattr(vid_result, 'error', 'unknown error')
            raise RuntimeError(f"Video generation failed in cycle {cycle_idx}: {error_msg}")

        shutil.copyfile(vid_result.video_path, video_path)
        logger.info(f"[SlidingStory] Generated video → {video_path}")
        # Extract last frame
        lastframe_name = f"lastframe_{cycle_idx:03d}.png"
        lastframe_path = keyframes_dir / lastframe_name
        _extract_last_frame_ffmpeg(video_path, lastframe_path)
        last_frame_path = lastframe_path
        final_video_paths.append(video_path)
        briq_data["paths"]["video"] = str(video_path)
        briq_data["paths"]["last_frame"] = str(lastframe_path)
        _write_briq_json(briqs_dir, cycle_idx, briq_data)

    # === Final assembly ===
    final_video_path = qodeyard_dir / "final_video.mp4"
    if not final_video_paths:
        raise RuntimeError("No video segments were generated; cannot assemble final video")
    # Concatenate videos via ffmpeg
    _concat_videos_ffmpeg(final_video_paths, final_video_path)
    logger.info(f"[SlidingStory] Final story video assembled → {final_video_path}")
    # Clean up temporary backend outputs
    try:
        shutil.rmtree(comfy_tmp)
    except Exception:
        pass
    return final_video_path
