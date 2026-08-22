#!/usr/bin/env python3
"""
image_metrics.py — Lightweight perceptual image metrics
═══════════════════════════════════════════════════════════════════════════════

Dependency-light quality heuristics used by the continuity guard and the
optional image-metrics diagnostics:

  - ``calculate_frame_similarity`` — structural/color/hash blend in [0, 1].
  - ``calculate_blur``             — blurriness in [0, 1]; higher = more blur.
  - ``calculate_entropy``          — Shannon entropy normalized to [0, 1].

All functions log (rather than print) on failure and return a safe numeric
fallback so callers never crash on a corrupt/odd frame.

Part of Visual FaQtory v0.9.5-beta
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def calculate_frame_similarity(
    image_a_path: str,
    image_b_path: str,
    *,
    size: Tuple[int, int] = (128, 72),
) -> float:
    """Compute a lightweight continuity score in the range [0, 1].

    Blends luminance correlation, edge correlation, RGB color similarity, and a
    small perceptual dHash agreement. 1.0 means "very similar", 0.0 means
    "wildly different".
    """
    try:
        with Image.open(image_a_path) as img_a, Image.open(image_b_path) as img_b:
            a_rgb = img_a.convert('RGB').resize(size, Image.Resampling.BICUBIC)
            b_rgb = img_b.convert('RGB').resize(size, Image.Resampling.BICUBIC)

            a = a_rgb.convert('L')
            b = b_rgb.convert('L')

            a_arr = np.asarray(a, dtype=np.float32) / 255.0
            b_arr = np.asarray(b, dtype=np.float32) / 255.0

            a_vec = a_arr.flatten() - float(a_arr.mean())
            b_vec = b_arr.flatten() - float(b_arr.mean())
            denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) + 1e-8
            luminance_corr = float(np.dot(a_vec, b_vec) / denom)

            a_edges = np.asarray(a.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
            b_edges = np.asarray(b.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
            ae_vec = a_edges.flatten() - float(a_edges.mean())
            be_vec = b_edges.flatten() - float(b_edges.mean())
            edge_denom = (np.linalg.norm(ae_vec) * np.linalg.norm(be_vec)) + 1e-8
            edge_corr = float(np.dot(ae_vec, be_vec) / edge_denom)

            a_rgb_arr = np.asarray(a_rgb.resize((64, 36), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            b_rgb_arr = np.asarray(b_rgb.resize((64, 36), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            color_mae = float(np.mean(np.abs(a_rgb_arr - b_rgb_arr)))
            color_score = max(0.0, min(1.0, 1.0 - color_mae))

            def _dhash_bits(img: Image.Image, hash_size: int = 8) -> np.ndarray:
                gray = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BICUBIC)
                arr = np.asarray(gray, dtype=np.float32)
                return (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()

            hash_a = _dhash_bits(a_rgb)
            hash_b = _dhash_bits(b_rgb)
            hash_score = float(np.mean(hash_a == hash_b))

            luminance_score = max(0.0, min(1.0, (luminance_corr + 1.0) / 2.0))
            edge_score = max(0.0, min(1.0, (edge_corr + 1.0) / 2.0))

            return float(
                (0.35 * luminance_score)
                + (0.25 * edge_score)
                + (0.25 * color_score)
                + (0.15 * hash_score)
            )
    except Exception as e:
        logger.warning(
            "Error calculating frame similarity for %s vs %s: %s",
            image_a_path,
            image_b_path,
            e,
        )
        return 0.0


def calculate_blur(image_path: str) -> float:
    """Return blurriness in [0, 1]; higher means more blur.

    Uses variance-of-Laplacian. A sharp image has high Laplacian variance and
    therefore a low blur score; a soft/blurry image has low variance and a high
    blur score. On error, returns 0.0 (interpreted as "no usable measurement")
    and logs a warning.
    """
    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert('L'), dtype=np.float64)

        # Discrete Laplacian kernel (3x3).
        laplacian = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
        )
        variance = float(np.var(laplacian))

        # Smoothly map variance → [0,1]. Variance around 100 (typical sharp
        # natural image) → ~0.5; a flat image (variance ~0) → 1.0.
        blur = 1.0 / (1.0 + variance / 100.0)
        return float(max(0.0, min(1.0, blur)))
    except Exception as e:
        logger.warning("Error calculating blur for %s: %s", image_path, e)
        return 0.0


def calculate_entropy(image_path: str) -> float:
    """Return Shannon entropy normalized to [0, 1]; higher = more detail/randomness."""
    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert('L'), dtype=np.uint8).ravel()

        hist = np.bincount(gray, minlength=256).astype(np.float64)
        total = float(hist.sum())
        if total <= 0:
            return 0.0
        probabilities = hist[hist > 0] / total
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        return max(0.0, min(1.0, entropy / 8.0))
    except Exception as e:
        logger.warning("Error calculating entropy for %s: %s", image_path, e)
        return 0.0


__all__ = ["calculate_frame_similarity", "calculate_blur", "calculate_entropy"]
