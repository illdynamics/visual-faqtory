#!/usr/bin/env python3
"""
base_folders.py - Base Folder Ingestion & File Selection
═══════════════════════════════════════════════════════════════════════════════

Handles automatic file selection from worqspace base folders:
  - worqspace/base_images/   (.png, .jpg, .jpeg, .webp)
  - worqspace/base_audio/   (.wav, .mp3, .flac, .aac, .m4a, .ogg)
  - worqspace/base_video/   (.mp4, .mov, .mkv, .webm)

Selection modes: newest | oldest | random | alphabetical
Random mode uses deterministic seed derived from run_id.

Does NOT break existing input modes — only activates if folders exist.

Part of QonQrete Visual FaQtory v0.5.6-beta
"""
import hashlib
import logging
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Recognized file extensions per folder type
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def select_base_files(
    worqspace_dir: Path,
    config: Dict[str, Any],
    run_id: str = "",
) -> Dict[str, Optional[Path]]:
    """
    Select one file from each base folder (if present and enabled).

    Args:
        worqspace_dir: Path to worqspace directory
        config: Full config dict
        run_id: Session/run ID for deterministic random seed

    Returns:
        Dict with keys 'base_image', 'base_audio', 'base_video',
        each mapping to a Path or None.
    """
    worqspace_dir = Path(worqspace_dir)
    bf_config = config.get("inputs", {}).get("base_folders", {})

    if not bf_config.get("enabled", True):
        logger.info("[BaseFolders] Base folders disabled in config")
        return {"base_image": None, "base_audio": None, "base_video": None}

    pick_mode = bf_config.get("pick_mode", "newest")
    random_seed = bf_config.get("random_seed")
    allow_empty = bf_config.get("allow_empty", True)

    # Derive deterministic seed from run_id if not explicitly set
    if random_seed is None and run_id:
        random_seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)

    # Folder paths (configurable with defaults)
    image_dir = worqspace_dir / bf_config.get("base_image_dir", "base_image")
    audio_dir = worqspace_dir / bf_config.get("base_audio_dir", "base_audio")
    video_dir = worqspace_dir / bf_config.get("base_video_dir", "base_video")

    results = {}

    results["base_image"] = _pick_file(
        image_dir, IMAGE_EXTENSIONS, pick_mode, random_seed, "base_image"
    )
    results["base_audio"] = _pick_file(
        audio_dir, AUDIO_EXTENSIONS, pick_mode, random_seed, "base_audio"
    )
    results["base_video"] = _pick_file(
        video_dir, VIDEO_EXTENSIONS, pick_mode, random_seed, "base_video"
    )

    if not allow_empty:
        for key, val in results.items():
            if val is None:
                folder = {"base_image": image_dir, "base_audio": audio_dir, "base_video": video_dir}[key]
                if folder.exists():
                    raise FileNotFoundError(
                        f"[BaseFolders] {key} folder exists but contains no valid files: {folder}"
                    )

    return results


def copy_selected_to_run(
    selected: Dict[str, Optional[Path]],
    run_dir: Path,
) -> Dict[str, Optional[Path]]:
    """
    Copy selected base files to the run's inputs directory.

    Args:
        selected: Dict from select_base_files()
        run_dir: Path to run directory (e.g. worqspace/runs/<run_id>/)

    Returns:
        Dict with paths to copies in the run directory
    """
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    copies = {}
    for key, src_path in selected.items():
        if src_path and src_path.exists():
            dest_name = f"selected_{key}{src_path.suffix}"
            dest_path = inputs_dir / dest_name
            shutil.copy2(src_path, dest_path)
            copies[key] = dest_path
            logger.info(f"[BaseFolders] Copied {key}: {src_path.name} → {dest_path}")
        else:
            copies[key] = None

    return copies


def _pick_file(
    folder: Path,
    valid_extensions: set,
    pick_mode: str,
    random_seed: Optional[int],
    label: str,
) -> Optional[Path]:
    """
    Pick a single file from a folder based on pick_mode.

    Args:
        folder: Directory to scan
        valid_extensions: Set of allowed file extensions (lowercase, with dot)
        pick_mode: 'newest' | 'oldest' | 'random' | 'alphabetical'
        random_seed: Seed for deterministic random selection
        label: Human-readable label for logging

    Returns:
        Path to selected file, or None if folder empty/missing
    """
    if not folder.exists():
        logger.debug(f"[BaseFolders] {label} folder not found: {folder}")
        return None

    candidates = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not candidates:
        logger.debug(f"[BaseFolders] {label} folder empty: {folder}")
        return None

    if pick_mode == "newest":
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        picked = candidates[0]
        reason = "newest by mtime"
    elif pick_mode == "oldest":
        candidates.sort(key=lambda f: f.stat().st_mtime)
        picked = candidates[0]
        reason = "oldest by mtime"
    elif pick_mode == "random":
        rng = random.Random(random_seed)
        picked = rng.choice(candidates)
        reason = f"random (seed={random_seed})"
    elif pick_mode == "alphabetical":
        candidates.sort(key=lambda f: f.name.lower())
        picked = candidates[0]
        reason = "alphabetical first"
    else:
        logger.warning(
            f"[BaseFolders] Unknown pick_mode '{pick_mode}', defaulting to newest"
        )
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        picked = candidates[0]
        reason = "newest (fallback)"

    logger.info(
        f"[BaseFolders] {label}: selected '{picked.name}' ({reason}) "
        f"from {len(candidates)} candidate(s)"
    )
    return picked


__all__ = [
    "select_base_files",
    "copy_selected_to_run",
    "IMAGE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
]
