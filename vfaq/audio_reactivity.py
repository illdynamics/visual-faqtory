#!/usr/bin/env python3
"""
audio_reactivity.py - BPM + Audio-Reactive Feature Extraction & Mapping
═══════════════════════════════════════════════════════════════════════════════

Provides:
  - Audio loading and analysis (BPM, beat grid, spectral features)
  - Per-frame feature extraction (RMS, spectral flux, centroid, etc.)
  - Band-split energy (sub/bass/lowmid/mid/high/air)
  - Beat grid generation with quantization
  - Reactive controller for cycle-based parameter mapping
  - Deterministic caching based on audio hash + config hash
  - BPM sync timing planner for cycle durations

Dependencies: librosa (primary), numpy
Fallback: graceful degradation if librosa is unavailable.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import json
import hashlib
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(
    path: Path, sr: int = 44100
) -> Tuple[Any, int]:
    """
    Load audio file, return (samples_array, sample_rate).
    Requires librosa.
    """
    import librosa
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    logger.info(
        f"[AudioReact] Loaded audio: {path.name} | "
        f"sr={sr_out} | duration={len(y)/sr_out:.2f}s | samples={len(y)}"
    )
    return y, sr_out


def audio_file_hash(path: Path) -> str:
    """SHA256 hash of the audio file for caching."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(audio_config: Dict[str, Any]) -> str:
    """Hash of audio reactivity config for cache invalidation."""
    raw = json.dumps(audio_config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# BPM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bpm(
    y: Any,
    sr: int,
    mode: str = "auto_then_override",
    manual_bpm: Optional[float] = None,
    doubletime_hint: bool = True,
) -> Dict[str, Any]:
    """
    Compute BPM with multiple modes.

    Modes:
      auto              - librosa tempo detection only
      manual            - use manual_bpm directly
      auto_then_override- detect auto, prefer manual if provided and close
      off               - skip BPM detection entirely

    Args:
        y:               Audio samples
        sr:              Sample rate
        mode:            BPM detection mode
        manual_bpm:      User-specified BPM override
        doubletime_hint: If auto detects ~half of manual, prefer 2x

    Returns:
        Dict with: bpm, bpm_confidence, source, raw_detected_bpm
    """
    result = {
        "bpm": 120.0,
        "bpm_confidence": 0.0,
        "source": "default",
        "raw_detected_bpm": None,
    }

    if mode == "off":
        result["source"] = "off"
        if manual_bpm:
            result["bpm"] = manual_bpm
            result["source"] = "manual"
        return result

    if mode == "manual":
        if manual_bpm:
            result["bpm"] = manual_bpm
            result["source"] = "manual"
            result["bpm_confidence"] = 1.0
        else:
            logger.warning("[AudioReact] BPM mode=manual but no manual_bpm set, using 120")
        return result

    # Auto detection
    detected_bpm = None
    confidence = 0.0
    try:
        import librosa
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return array; extract scalar
        if hasattr(tempo, '__len__'):
            detected_bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            detected_bpm = float(tempo)
        confidence = min(1.0, len(beat_frames) / 50.0)  # rough confidence proxy
        result["raw_detected_bpm"] = detected_bpm
        logger.info(
            f"[AudioReact] Auto BPM detected: {detected_bpm:.1f} "
            f"(confidence ~{confidence:.2f}, {len(beat_frames)} beats)"
        )
    except Exception as e:
        logger.warning(f"[AudioReact] Auto BPM detection failed: {e}")
        detected_bpm = 120.0

    if mode == "auto":
        result["bpm"] = detected_bpm
        result["bpm_confidence"] = confidence
        result["source"] = "auto"
        return result

    # auto_then_override
    if manual_bpm:
        # Doubletime hint: if auto detects ~half of manual, prefer manual
        if doubletime_hint and detected_bpm and manual_bpm > 0:
            ratio = manual_bpm / detected_bpm
            if 1.8 < ratio < 2.2:
                logger.info(
                    f"[AudioReact] Doubletime hint: auto={detected_bpm:.1f}, "
                    f"manual={manual_bpm}, using manual (ratio {ratio:.2f})"
                )
                result["bpm"] = manual_bpm
                result["source"] = "manual_doubletime"
                result["bpm_confidence"] = 1.0
                return result

        result["bpm"] = manual_bpm
        result["source"] = "manual_override"
        result["bpm_confidence"] = 1.0
    else:
        result["bpm"] = detected_bpm
        result["source"] = "auto"
        result["bpm_confidence"] = confidence

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BEAT GRID GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

QUANTIZE_MAP = {
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}


def compute_beat_grid(
    bpm: float,
    sr: int,
    duration: float,
    quantize: str = "1/16",
    beats_per_bar: int = 4,
) -> Dict[str, Any]:
    """
    Generate a beat grid from BPM.

    Args:
        bpm:           Beats per minute
        sr:            Sample rate
        duration:      Total audio duration in seconds
        quantize:      Grid quantization ('1/4', '1/8', '1/16', '1/32')
        beats_per_bar: Beats per bar (default 4/4 time)

    Returns:
        Dict with: beat_times, bar_times, subdivision_times, bpm, quantize, total_bars
    """
    beat_duration = 60.0 / bpm  # seconds per beat
    bar_duration = beat_duration * beats_per_bar
    subdiv_factor = QUANTIZE_MAP.get(quantize, 0.25)
    subdiv_duration = beat_duration * subdiv_factor

    # Generate timestamps
    beat_times = []
    bar_times = []
    subdivision_times = []

    t = 0.0
    beat_count = 0
    while t < duration:
        beat_times.append(t)
        if beat_count % beats_per_bar == 0:
            bar_times.append(t)
        beat_count += 1
        t += beat_duration

    t = 0.0
    while t < duration:
        subdivision_times.append(t)
        t += subdiv_duration

    total_bars = len(bar_times)

    logger.info(
        f"[AudioReact] Beat grid: bpm={bpm:.1f}, bars={total_bars}, "
        f"beats={len(beat_times)}, subdivisions={len(subdivision_times)} ({quantize})"
    )

    return {
        "bpm": bpm,
        "beats_per_bar": beats_per_bar,
        "quantize": quantize,
        "beat_duration": beat_duration,
        "bar_duration": bar_duration,
        "total_bars": total_bars,
        "beat_times": beat_times,
        "bar_times": bar_times,
        "subdivision_times": subdivision_times,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_BANDS = {
    "sub": [20, 60],
    "bass": [60, 180],
    "lowmid": [180, 600],
    "mid": [600, 2000],
    "high": [2000, 6000],
    "air": [6000, 16000],
}


def compute_features(
    y: Any,
    sr: int,
    hop_length: int = 512,
    bands: Optional[Dict[str, List[int]]] = None,
    features_config: Optional[Dict[str, bool]] = None,
    smoothing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract per-frame audio features.

    Features extracted:
      - rms: Root mean square energy
      - spectral_flux: Rate of spectral change
      - centroid: Spectral centroid (brightness)
      - rolloff: Spectral rolloff frequency
      - onset_strength: Onset detection envelope
      - onsets: Detected onset timestamps
      - band_rms: Per-band RMS energy (sub, bass, lowmid, mid, high, air)

    All arrays are per-frame (hop_length granularity).

    Args:
        y:                Audio samples
        sr:               Sample rate
        hop_length:       Hop length for STFT
        bands:            Frequency band definitions
        features_config:  Which features to extract
        smoothing_config: Smoothing parameters

    Returns:
        Dict with feature arrays and metadata
    """
    import numpy as np
    import librosa

    if bands is None:
        bands = DEFAULT_BANDS
    if features_config is None:
        features_config = {
            "rms": True, "spectral_flux": True, "centroid": True,
            "rolloff": True, "onset_strength": True, "onsets": True,
        }
    if smoothing_config is None:
        smoothing_config = {"enabled": True, "method": "ema", "ema_alpha": 0.25}

    n_frames = 1 + len(y) // hop_length
    result = {"n_frames": n_frames, "hop_length": hop_length, "sr": sr}

    # STFT for band-split energy
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * (S.shape[0] - 1))

    # Band RMS
    band_rms = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_energy = np.sqrt(np.mean(S[mask, :] ** 2, axis=0))
            band_rms[band_name] = band_energy.tolist()
        else:
            band_rms[band_name] = [0.0] * S.shape[1]
    result["band_rms"] = band_rms

    # Global RMS
    if features_config.get("rms", True):
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        result["rms"] = rms.tolist()

    # Spectral centroid
    if features_config.get("centroid", True):
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        result["centroid"] = centroid.tolist()

    # Spectral rolloff
    if features_config.get("rolloff", True):
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        result["rolloff"] = rolloff.tolist()

    # Spectral flux
    if features_config.get("spectral_flux", True):
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        result["spectral_flux"] = flux.tolist()

    # Onset strength envelope + onset times
    if features_config.get("onset_strength", True):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        result["onset_strength"] = onset_env.tolist()

    if features_config.get("onsets", True):
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        result["onset_times"] = onset_times.tolist()
        result["onset_count"] = len(onset_times)

    # Apply smoothing
    if smoothing_config.get("enabled", True):
        method = smoothing_config.get("method", "ema")
        alpha = smoothing_config.get("ema_alpha", 0.25)
        for key in ["rms", "centroid", "rolloff", "spectral_flux", "onset_strength"]:
            if key in result and isinstance(result[key], list):
                result[key] = _smooth(result[key], method, alpha)
        for band_name in band_rms:
            band_rms[band_name] = _smooth(band_rms[band_name], method, alpha)

    return result


def _smooth(data: list, method: str, alpha: float) -> list:
    """Apply smoothing to a feature array."""
    if not data or method == "none":
        return data

    if method == "ema":
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1.0 - alpha) * smoothed[-1])
        return smoothed
    elif method == "median":
        import numpy as np
        kernel = 5
        arr = np.array(data)
        padded = np.pad(arr, kernel // 2, mode="edge")
        result = []
        for i in range(len(arr)):
            window = padded[i : i + kernel]
            result.append(float(np.median(window)))
        return result

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# REACTIVE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class ReactiveController:
    """
    Exposes audio features for a given time point or cycle segment.

    Usage:
        ctrl = ReactiveController(features, beat_grid, sr, hop_length)
        val = ctrl.get_feature_at("bass.rms", t=2.5)
        phase = ctrl.get_beat_phase(t=2.5)
        is_on_beat = ctrl.is_beat(t=2.5, tolerance_ms=50)
    """

    def __init__(
        self,
        features: Dict[str, Any],
        beat_grid: Dict[str, Any],
        sr: int = 44100,
        hop_length: int = 512,
    ):
        self.features = features
        self.beat_grid = beat_grid
        self.sr = sr
        self.hop_length = hop_length
        self.frame_duration = hop_length / sr

    def time_to_frame(self, t: float) -> int:
        """Convert time in seconds to frame index."""
        frame = int(t / self.frame_duration)
        n_frames = self.features.get("n_frames", 1)
        return max(0, min(frame, n_frames - 1))

    def get_feature_at(self, feature_path: str, t: float) -> float:
        """
        Get feature value at time t.

        feature_path examples:
          'rms', 'centroid', 'bass.rms', 'high.rms', 'onset_strength'
        """
        frame = self.time_to_frame(t)

        if "." in feature_path:
            band, feat = feature_path.split(".", 1)
            if feat == "rms":
                band_data = self.features.get("band_rms", {}).get(band, [])
                if frame < len(band_data):
                    return band_data[frame]
                return 0.0

        data = self.features.get(feature_path, [])
        if isinstance(data, list) and frame < len(data):
            return data[frame]
        return 0.0

    def get_segment_stats(
        self, start_time: float, end_time: float
    ) -> Dict[str, float]:
        """
        Compute summary statistics for a time segment.

        Returns:
            Dict with avg/peak values for key features
        """
        import numpy as np

        start_frame = self.time_to_frame(start_time)
        end_frame = self.time_to_frame(end_time)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        stats = {}

        # Global features
        for key in ["rms", "centroid", "spectral_flux", "onset_strength"]:
            data = self.features.get(key, [])
            if data:
                segment = data[start_frame:end_frame]
                if segment:
                    arr = np.array(segment)
                    stats[f"avg_{key}"] = float(np.mean(arr))
                    stats[f"peak_{key}"] = float(np.max(arr))

        # Band energies
        for band_name, band_data in self.features.get("band_rms", {}).items():
            if band_data:
                segment = band_data[start_frame:end_frame]
                if segment:
                    arr = np.array(segment)
                    stats[f"avg_{band_name}_rms"] = float(np.mean(arr))
                    stats[f"peak_{band_name}_rms"] = float(np.max(arr))

        # Onset/transient count in segment
        onset_times = self.features.get("onset_times", [])
        transient_count = sum(
            1 for ot in onset_times if start_time <= ot < end_time
        )
        stats["transient_count"] = transient_count

        return stats

    def get_beat_phase(self, t: float) -> float:
        """Returns 0.0..1.0 within the current beat."""
        beat_dur = self.beat_grid.get("beat_duration", 0.5)
        if beat_dur <= 0:
            return 0.0
        phase = (t % beat_dur) / beat_dur
        return phase

    def get_bar_phase(self, t: float) -> float:
        """Returns 0.0..1.0 within the current bar."""
        bar_dur = self.beat_grid.get("bar_duration", 2.0)
        if bar_dur <= 0:
            return 0.0
        return (t % bar_dur) / bar_dur

    def is_beat(self, t: float, tolerance_ms: float = 50.0) -> bool:
        """Check if time t is near a beat boundary."""
        tol = tolerance_ms / 1000.0
        for bt in self.beat_grid.get("beat_times", []):
            if abs(t - bt) < tol:
                return True
        return False

    def is_bar(self, t: float, tolerance_ms: float = 50.0) -> bool:
        """Check if time t is near a bar boundary."""
        tol = tolerance_ms / 1000.0
        for bt in self.beat_grid.get("bar_times", []):
            if abs(t - bt) < tol:
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE TIMING PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cycle_timing(
    bpm: float,
    bars_per_cycle: int = 8,
    total_cycles: int = 5,
    beats_per_bar: int = 4,
) -> Dict[str, Any]:
    """
    Plan cycle durations synced to BPM.

    Args:
        bpm:            Beats per minute
        bars_per_cycle: How many bars per generation cycle
        total_cycles:   Number of cycles planned
        beats_per_bar:  Time signature (default 4/4)

    Returns:
        Dict with: cycle_duration_sec, total_duration_sec, cycles list with start/end times
    """
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * beats_per_bar
    cycle_dur = bar_dur * bars_per_cycle
    total_dur = cycle_dur * total_cycles

    cycles = []
    for i in range(total_cycles):
        cycles.append({
            "cycle_index": i,
            "start_time": cycle_dur * i,
            "end_time": cycle_dur * (i + 1),
            "bar_start": bars_per_cycle * i,
            "bar_end": bars_per_cycle * (i + 1),
        })

    logger.info(
        f"[AudioReact] Cycle timing: bpm={bpm:.1f}, {bars_per_cycle} bars/cycle, "
        f"cycle_dur={cycle_dur:.2f}s, total={total_dur:.2f}s"
    )

    return {
        "bpm": bpm,
        "bars_per_cycle": bars_per_cycle,
        "beats_per_bar": beats_per_bar,
        "cycle_duration_sec": cycle_dur,
        "total_duration_sec": total_dur,
        "cycles": cycles,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER MAPPING (Audio → Visual)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_audio_mapping(
    segment_stats: Dict[str, float],
    mapping_config: Dict[str, Any],
    cycle_index: int = 0,
    beat_grid: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Map audio segment statistics to visual parameters.

    Mapping config format:
      prompt_modifiers:
        intensity:
          source: "bass.rms"     ->  uses avg_bass_rms
          curve: "linear"
          min: 0.0
          max: 1.0
          inject_as: "intensity"
      seed:
        mode: "bar_step"
        bar_step_amount: 17
      transitions:
        cut_on: "bar"
        crossfade_ms: 250

    Returns:
        Dict with: prompt_additions (str), seed_offset (int), modulated_params (dict)
    """
    result = {
        "prompt_additions": "",
        "seed_offset": 0,
        "modulated_params": {},
    }

    if not mapping_config or not mapping_config.get("enabled", True):
        return result

    # Prompt modifiers
    prompt_parts = []
    modifiers_cfg = mapping_config.get("prompt_modifiers", {})
    for mod_name, mod_cfg in modifiers_cfg.items():
        source = mod_cfg.get("source", "rms")
        curve = mod_cfg.get("curve", "linear")
        val_min = mod_cfg.get("min", 0.0)
        val_max = mod_cfg.get("max", 1.0)
        inject_as = mod_cfg.get("inject_as", mod_name)

        # Get raw value from stats
        stat_key = f"avg_{source.replace('.', '_')}"
        raw_val = segment_stats.get(stat_key, 0.0)

        # Normalize to 0..1 range (approximate)
        normalized = min(1.0, max(0.0, raw_val))

        # Apply curve
        if curve == "exp":
            normalized = normalized ** 2
        elif curve == "sqrt":
            normalized = math.sqrt(normalized)
        elif curve == "log":
            normalized = math.log1p(normalized * 10) / math.log1p(10)

        # Scale to min/max
        mapped = val_min + (val_max - val_min) * normalized
        result["modulated_params"][inject_as] = mapped
        prompt_parts.append(f"{inject_as}:{mapped:.2f}")

    if prompt_parts:
        result["prompt_additions"] = ", ".join(prompt_parts)

    # Seed modulation
    seed_cfg = mapping_config.get("seed", {})
    seed_mode = seed_cfg.get("mode", "none")
    if seed_mode == "bar_step":
        step = seed_cfg.get("bar_step_amount", 17)
        result["seed_offset"] = cycle_index * step
    elif seed_mode == "beat_step":
        step = seed_cfg.get("beat_step_amount", 7)
        result["seed_offset"] = cycle_index * step
    elif seed_mode == "energy_jitter":
        energy = segment_stats.get("avg_rms", 0.0)
        result["seed_offset"] = int(energy * 100) % 100

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_cache_path(
    worqspace_dir: Path, file_hash: str, cfg_hash: str
) -> Path:
    """Return cache directory for given audio + config combination."""
    return Path(worqspace_dir) / "cache" / "audio" / file_hash[:16] / cfg_hash


def save_analysis_cache(
    cache_dir: Path,
    analysis: Dict[str, Any],
    beat_grid: Dict[str, Any],
    features: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist analysis artifacts to cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str)
    )
    (cache_dir / "beat_grid.json").write_text(
        json.dumps(beat_grid, indent=2, default=str)
    )
    if features:
        # Save features as JSON (no numpy dependency for loading)
        features_serializable = {
            k: v for k, v in features.items()
            if not k.startswith("_")
        }
        (cache_dir / "features.json").write_text(
            json.dumps(features_serializable, indent=2, default=str)
        )

    logger.info(f"[AudioReact] Analysis cached to {cache_dir}")


def load_analysis_cache(
    cache_dir: Path,
) -> Optional[Tuple[Dict, Dict, Optional[Dict]]]:
    """Load analysis from cache. Returns (analysis, beat_grid, features) or None."""
    analysis_path = cache_dir / "analysis.json"
    grid_path = cache_dir / "beat_grid.json"

    if not analysis_path.exists() or not grid_path.exists():
        return None

    analysis = json.loads(analysis_path.read_text())
    beat_grid = json.loads(grid_path.read_text())
    features = None

    features_path = cache_dir / "features.json"
    if features_path.exists():
        features = json.loads(features_path.read_text())

    logger.info(f"[AudioReact] Cache hit: {cache_dir}")
    return analysis, beat_grid, features


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_audio_analysis(
    audio_path: Path,
    worqspace_dir: Path,
    audio_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], ReactiveController]:
    """
    Run the full audio analysis pipeline with caching.

    Args:
        audio_path:    Path to audio file
        worqspace_dir: Worqspace root (for cache)
        audio_config:  audio_reactivity section from config.yaml

    Returns:
        (analysis_meta, beat_grid, features, reactive_controller)
    """
    sr = audio_config.get("sample_rate", 44100)
    hop_length = audio_config.get("hop_length", 512)

    # Check cache
    file_hash_val = audio_file_hash(audio_path)
    cfg_hash_val = config_hash(audio_config)
    cache_dir = get_cache_path(worqspace_dir, file_hash_val, cfg_hash_val)

    cached = load_analysis_cache(cache_dir)
    if cached:
        analysis, beat_grid, features = cached
        if features:
            ctrl = ReactiveController(features, beat_grid, sr, hop_length)
            return analysis, beat_grid, features, ctrl

    # Load audio
    y, sr = load_audio(audio_path, sr=sr)
    duration = len(y) / sr

    # BPM
    bpm_result = compute_bpm(
        y, sr,
        mode=audio_config.get("bpm_mode", "auto_then_override"),
        manual_bpm=audio_config.get("bpm_manual"),
        doubletime_hint=audio_config.get("bpm_doubletime_hint", True),
    )

    # Beat grid
    grid_cfg = audio_config.get("beat_grid", {})
    beat_grid = compute_beat_grid(
        bpm=bpm_result["bpm"],
        sr=sr,
        duration=duration,
        quantize=grid_cfg.get("quantize", "1/16"),
    )

    # Features
    bands = audio_config.get("bands", DEFAULT_BANDS)
    features_cfg = audio_config.get("features", {})
    smoothing_cfg = audio_config.get("smoothing", {})

    features = compute_features(
        y, sr,
        hop_length=hop_length,
        bands=bands,
        features_config=features_cfg,
        smoothing_config=smoothing_cfg,
    )

    # Analysis metadata
    analysis = {
        "audio_file": str(audio_path),
        "audio_hash": file_hash_val,
        "config_hash": cfg_hash_val,
        "duration": duration,
        "sample_rate": sr,
        "hop_length": hop_length,
        **bpm_result,
    }

    # Cache results
    save_analysis_cache(cache_dir, analysis, beat_grid, features)

    # Build controller
    ctrl = ReactiveController(features, beat_grid, sr, hop_length)

    return analysis, beat_grid, features, ctrl


__all__ = [
    "load_audio",
    "audio_file_hash",
    "compute_bpm",
    "compute_beat_grid",
    "compute_features",
    "ReactiveController",
    "compute_cycle_timing",
    "apply_audio_mapping",
    "run_audio_analysis",
    "save_analysis_cache",
    "load_analysis_cache",
    "DEFAULT_BANDS",
]
