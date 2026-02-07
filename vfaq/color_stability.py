#!/usr/bin/env python3
"""
color_stability.py - Long-Run Color Stability Controller
═══════════════════════════════════════════════════════════════════════════════

Prevents diffusion feedback collapse in long-running Visual FaQtory sessions.

PROBLEM: Iterative diffusion (turbo frame loops, stream continuation, chained
cycles) accumulates colour drift. After 100+ frames the image can converge to
a single-hue low-saturation "green blob". This module provides CPU-side,
frame-safe corrections that keep the output visually consistent without
requiring model changes or base image reloads.

Mechanisms:

  1. **Palette Anchoring** (LAB colour space)
     - Captures the first output frame as an "anchor".
     - On every subsequent frame, matches the mean and standard deviation of
       the A and B channels to the anchor, blending by `strength`.
     - Operates in CIELAB so hue and saturation corrections are perceptually
       uniform.

  2. **Collapse Detection**
     - Monitors three metrics frame-over-frame:
         • Mean saturation (HSV S-channel)
         • Channel dominance (max single-channel ratio in the hue histogram)
         • Edge energy (Sobel magnitude mean)
     - If all three cross their thresholds for `consecutive_frames` frames,
       the system flags a collapse event.

  3. **Collapse Mitigation**
     - Reduces generation CFG by a configured factor (signalled to caller).
     - Reduces seed drift by a configured factor (signalled to caller).
     - Injects micro-noise (Gaussian, sigma configurable) to break the
       attractor basin.
     - Re-anchors the palette from the mitigated frame.

All operations are CPU-side, use numpy/cv2 if available, and fall back to
PIL-only if OpenCV is missing. The controller never blocks frame output in
live mode: if the time budget is exceeded it skips correction silently.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports — we degrade gracefully
try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None


def _ensure_numpy():
    if np is None:
        raise ImportError("numpy is required for color_stability")


def _pil_to_np(img) -> "np.ndarray":
    """Convert PIL Image to numpy HxWxC uint8 array."""
    _ensure_numpy()
    return np.array(img.convert("RGB"))


def _np_to_pil(arr: "np.ndarray"):
    """Convert numpy HxWxC uint8 array to PIL Image."""
    if Image is None:
        raise ImportError("Pillow is required for color_stability")
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB CONVERSION (pure numpy, no OpenCV dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _srgb_to_linear(c: "np.ndarray") -> "np.ndarray":
    """Convert sRGB [0,1] to linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: "np.ndarray") -> "np.ndarray":
    """Convert linear RGB to sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 1e-10), 1.0 / 2.4) - 0.055)


def rgb_to_lab(img_uint8: "np.ndarray") -> "np.ndarray":
    """Convert uint8 RGB image to CIELAB float32 (L 0-100, a/b approx ±128)."""
    _ensure_numpy()
    rgb = img_uint8.astype(np.float32) / 255.0
    linear = _srgb_to_linear(rgb)
    # sRGB → XYZ (D65 illuminant)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ mat.T
    # Normalise to D65 white point
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883
    # f(t) for LAB
    delta = 6.0 / 29.0
    t_thresh = delta ** 3
    f = np.where(xyz > t_thresh, np.power(np.maximum(xyz, 1e-10), 1.0 / 3.0),
                 xyz / (3.0 * delta * delta) + 4.0 / 29.0)
    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: "np.ndarray") -> "np.ndarray":
    """Convert CIELAB float32 to uint8 RGB."""
    _ensure_numpy()
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    t_thresh = delta

    def finv(f_val):
        return np.where(f_val > t_thresh, f_val ** 3, 3.0 * delta * delta * (f_val - 4.0 / 29.0))

    x = finv(fx) * 0.95047
    y = finv(fy) * 1.00000
    z = finv(fz) * 1.08883
    xyz = np.stack([x, y, z], axis=-1)
    # XYZ → linear sRGB
    mat_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32)
    linear = xyz @ mat_inv.T
    linear = np.clip(linear, 0.0, None)
    srgb = _linear_to_srgb(linear)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# HSV CONVERSION (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def _rgb_to_hsv(rgb_uint8: "np.ndarray") -> "np.ndarray":
    """Convert uint8 RGB to float32 HSV (H 0-360, S 0-1, V 0-1)."""
    _ensure_numpy()
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # Hue
    h = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)
    return np.stack([h, s, cmax], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SOBEL EDGE ENERGY (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def _edge_energy(gray: "np.ndarray") -> float:
    """Compute mean Sobel gradient magnitude on a grayscale image."""
    _ensure_numpy()
    # Sobel 3x3 via convolution
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    # Horizontal Sobel
    gx[1:-1, 1:-1] = (
        -1 * gray[:-2, :-2] + 0 * gray[:-2, 1:-1] + 1 * gray[:-2, 2:]
        - 2 * gray[1:-1, :-2] + 0 * gray[1:-1, 1:-1] + 2 * gray[1:-1, 2:]
        - 1 * gray[2:, :-2] + 0 * gray[2:, 1:-1] + 1 * gray[2:, 2:]
    )
    # Vertical Sobel
    gy[1:-1, 1:-1] = (
        -1 * gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - 1 * gray[:-2, 2:]
        + 0 * gray[1:-1, :-2] + 0 * gray[1:-1, 1:-1] + 0 * gray[1:-1, 2:]
        + 1 * gray[2:, :-2] + 2 * gray[2:, 1:-1] + 1 * gray[2:, 2:]
    )
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag))


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class StabilityController:
    """
    CPU-side color stability and collapse prevention for long-running diffusion.

    Usage:
        ctrl = StabilityController(config['stability'])
        # On each frame:
        frame_np = ctrl.process_frame(frame_np)  # returns corrected frame
        mods = ctrl.get_generation_modifiers()    # returns cfg/seed/noise adjustments
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled: bool = bool(config.get("enabled", True))
        self.anchor_mode: str = str(config.get("anchor", "first_frame"))
        self.method: str = str(config.get("method", "lab_palette"))
        self.strength: float = float(config.get("strength", 0.6))
        self.every_n: int = max(1, int(config.get("every_n_frames", 1)))

        # Collapse detection config
        cd = config.get("collapse_detection", {})
        self.cd_enabled: bool = bool(cd.get("enabled", True))
        self.cd_sat_floor: float = float(cd.get("sat_floor", 0.08))
        self.cd_hue_dom_ratio: float = float(cd.get("hue_dom_ratio", 0.72))
        self.cd_edge_floor: float = float(cd.get("edge_floor", 3.0))
        self.cd_consecutive: int = int(cd.get("consecutive_frames", 12))

        # Mitigation config
        mit = config.get("mitigation", {})
        self.mit_reduce_cfg: float = float(mit.get("reduce_cfg", 0.15))
        self.mit_reduce_seed_drift: float = float(mit.get("reduce_seed_drift", 0.5))
        self.mit_micro_noise_sigma: float = float(mit.get("micro_noise_sigma", 0.005))

        # Internal state
        self._anchor_lab: Optional["np.ndarray"] = None
        self._anchor_mean_a: float = 0.0
        self._anchor_mean_b: float = 0.0
        self._anchor_std_a: float = 1.0
        self._anchor_std_b: float = 1.0
        self._frame_count: int = 0
        self._collapse_streak: int = 0
        self._collapse_active: bool = False

        # Generation modifier signals
        self._cfg_modifier: float = 0.0
        self._seed_drift_modifier: float = 0.0

        if not self.enabled:
            logger.info("[Stability] Controller disabled by config")
        else:
            logger.info(
                f"[Stability] Enabled: method={self.method}, strength={self.strength}, "
                f"collapse_detection={'on' if self.cd_enabled else 'off'}"
            )

    def process_frame(
        self,
        frame: "np.ndarray",
        time_budget_ms: float = 0,
    ) -> "np.ndarray":
        """
        Process a frame through the stability pipeline.

        Args:
            frame: HxWxC uint8 RGB numpy array
            time_budget_ms: Maximum allowed processing time in ms.
                            0 = unlimited. If exceeded, returns frame unmodified.

        Returns:
            Corrected frame as HxWxC uint8 RGB numpy array.
        """
        if not self.enabled or np is None:
            return frame

        self._frame_count += 1
        start = time.monotonic()

        try:
            # Set anchor on first frame
            if self._anchor_lab is None:
                self._set_anchor(frame)
                return frame

            # Skip frames based on every_n
            if self._frame_count % self.every_n != 0:
                return frame

            # Check time budget (skip if already exceeded)
            if time_budget_ms > 0:
                elapsed_ms = (time.monotonic() - start) * 1000
                if elapsed_ms > time_budget_ms * 0.5:
                    return frame

            # Run collapse detection
            if self.cd_enabled:
                self._detect_collapse(frame)

            # Apply palette anchoring
            corrected = self._apply_palette_anchoring(frame)

            # Apply micro-noise if collapse is active
            if self._collapse_active:
                corrected = self._apply_micro_noise(corrected)

            # Time budget check before returning
            if time_budget_ms > 0:
                elapsed_ms = (time.monotonic() - start) * 1000
                if elapsed_ms > time_budget_ms:
                    logger.debug("[Stability] Time budget exceeded, returning uncorrected frame")
                    return frame

            return corrected

        except Exception as e:
            logger.debug(f"[Stability] Frame processing error: {e}")
            return frame

    def process_pil_frame(self, pil_img, time_budget_ms: float = 0):
        """Process a PIL Image and return a PIL Image."""
        if not self.enabled or np is None or Image is None:
            return pil_img
        arr = _pil_to_np(pil_img)
        result = self.process_frame(arr, time_budget_ms)
        return _np_to_pil(result)

    def process_frame_bytes(self, jpg_bytes: bytes, time_budget_ms: float = 0) -> bytes:
        """Process JPEG bytes and return corrected JPEG bytes."""
        if not self.enabled or np is None or Image is None:
            return jpg_bytes
        try:
            import io
            pil_img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            arr = _pil_to_np(pil_img)
            result = self.process_frame(arr, time_budget_ms)
            pil_out = _np_to_pil(result)
            buf = io.BytesIO()
            pil_out.save(buf, format="JPEG", quality=92)
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"[Stability] JPEG processing error: {e}")
            return jpg_bytes

    def get_generation_modifiers(self) -> Dict[str, float]:
        """
        Return modifier signals for the generation backend.

        Returns:
            dict with:
                cfg_reduce: float (0.0 = no change, positive = reduce by this amount)
                seed_drift_reduce: float (0.0 = no change, positive = reduce multiplier)
                collapse_active: bool
        """
        return {
            "cfg_reduce": self._cfg_modifier,
            "seed_drift_reduce": self._seed_drift_modifier,
            "collapse_active": self._collapse_active,
        }

    def reset_anchor(self, frame: Optional["np.ndarray"] = None) -> None:
        """Reset the anchor to a new frame or clear it entirely."""
        if frame is not None:
            self._set_anchor(frame)
        else:
            self._anchor_lab = None
            self._anchor_mean_a = 0.0
            self._anchor_mean_b = 0.0
            self._anchor_std_a = 1.0
            self._anchor_std_b = 1.0
        self._collapse_streak = 0
        self._collapse_active = False
        self._cfg_modifier = 0.0
        self._seed_drift_modifier = 0.0
        logger.info("[Stability] Anchor reset")

    # ─── Private methods ─────────────────────────────────────────────────

    def _set_anchor(self, frame: "np.ndarray") -> None:
        """Compute and store anchor statistics from a frame."""
        lab = rgb_to_lab(frame)
        self._anchor_lab = lab
        a_chan = lab[:, :, 1].flatten()
        b_chan = lab[:, :, 2].flatten()
        self._anchor_mean_a = float(np.mean(a_chan))
        self._anchor_mean_b = float(np.mean(b_chan))
        self._anchor_std_a = max(float(np.std(a_chan)), 0.01)
        self._anchor_std_b = max(float(np.std(b_chan)), 0.01)
        logger.info(
            f"[Stability] Anchor set: A_mean={self._anchor_mean_a:.2f}, "
            f"B_mean={self._anchor_mean_b:.2f}"
        )

    def _apply_palette_anchoring(self, frame: "np.ndarray") -> "np.ndarray":
        """
        Match the A and B channel statistics of the frame to the anchor.

        Uses mean/std matching in CIELAB space, then blends with the
        original by `self.strength`.
        """
        lab = rgb_to_lab(frame)
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]

        # Current statistics
        cur_mean_a = float(np.mean(a_chan))
        cur_mean_b = float(np.mean(b_chan))
        cur_std_a = max(float(np.std(a_chan)), 0.01)
        cur_std_b = max(float(np.std(b_chan)), 0.01)

        # Match mean and std to anchor
        new_a = (a_chan - cur_mean_a) * (self._anchor_std_a / cur_std_a) + self._anchor_mean_a
        new_b = (b_chan - cur_mean_b) * (self._anchor_std_b / cur_std_b) + self._anchor_mean_b

        # Blend with original
        s = self.strength
        lab[:, :, 1] = a_chan * (1.0 - s) + new_a * s
        lab[:, :, 2] = b_chan * (1.0 - s) + new_b * s

        return lab_to_rgb(lab)

    def _detect_collapse(self, frame: "np.ndarray") -> None:
        """Detect color collapse using saturation, dominance and edge metrics."""
        # HSV for saturation
        hsv = _rgb_to_hsv(frame)
        mean_sat = float(np.mean(hsv[:, :, 1]))

        # Hue dominance: bin hues into 12 sectors, check if one dominates
        hues = hsv[:, :, 0].flatten()
        hist, _ = np.histogram(hues, bins=12, range=(0, 360))
        total = hist.sum()
        dom_ratio = float(hist.max()) / max(total, 1)

        # Edge energy
        gray = np.mean(frame.astype(np.float32), axis=2)
        edge = _edge_energy(gray)

        # Check collapse criteria
        sat_collapsed = mean_sat < self.cd_sat_floor
        hue_collapsed = dom_ratio > self.cd_hue_dom_ratio
        edge_collapsed = edge < self.cd_edge_floor

        if sat_collapsed and hue_collapsed and edge_collapsed:
            self._collapse_streak += 1
        else:
            self._collapse_streak = max(0, self._collapse_streak - 1)

        # Trigger collapse mitigation
        if self._collapse_streak >= self.cd_consecutive and not self._collapse_active:
            self._collapse_active = True
            self._cfg_modifier = self.mit_reduce_cfg
            self._seed_drift_modifier = self.mit_reduce_seed_drift
            # Re-anchor from current (mitigated) frame
            self._set_anchor(frame)
            logger.warning(
                f"[Stability] COLLAPSE DETECTED after {self._collapse_streak} frames "
                f"(sat={mean_sat:.3f}, dom={dom_ratio:.3f}, edge={edge:.2f}). "
                f"Mitigating: cfg-={self._cfg_modifier}, seed_drift×{1-self._seed_drift_modifier}"
            )
        elif self._collapse_streak == 0 and self._collapse_active:
            # Recovery
            self._collapse_active = False
            self._cfg_modifier = 0.0
            self._seed_drift_modifier = 0.0
            logger.info("[Stability] Collapse recovery — metrics normalised")

    def _apply_micro_noise(self, frame: "np.ndarray") -> "np.ndarray":
        """Inject small Gaussian noise to break attractor basins."""
        sigma = self.mit_micro_noise_sigma * 255.0
        if sigma <= 0:
            return frame
        noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_stability_controller(config: Dict[str, Any]) -> Optional[StabilityController]:
    """Create a StabilityController from the top-level config dict.

    Returns None if the stability section is absent or disabled, or if numpy
    is not available.
    """
    stab_cfg = config.get("stability", {})
    if not stab_cfg or not stab_cfg.get("enabled", False):
        return None
    if np is None:
        logger.warning("[Stability] numpy not available; stability controller disabled")
        return None
    return StabilityController(stab_cfg)


__all__ = [
    "StabilityController",
    "create_stability_controller",
    "rgb_to_lab",
    "lab_to_rgb",
]
