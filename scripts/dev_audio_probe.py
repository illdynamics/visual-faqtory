#!/usr/bin/env python3
"""
dev_audio_probe.py - Audio Analysis Diagnostic Tool
═══════════════════════════════════════════════════════════════════════════════

Quick diagnostic for audio files — prints BPM, band energies, onset count,
first 10 beat times. Useful for tuning audio_reactivity config.

Usage:
    python scripts/dev_audio_probe.py worqspace/base_audio/track.wav
    python scripts/dev_audio_probe.py track.mp3 --bpm 174

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Audio probe for Visual FaQtory")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    args = parser.parse_args()

    try:
        import librosa
        import numpy as np
    except ImportError:
        print("ERROR: librosa and numpy are required.")
        print("Install with: pip install librosa numpy")
        sys.exit(1)

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from vfaq.audio_reactivity import (
        load_audio, compute_bpm, compute_beat_grid, compute_features,
        DEFAULT_BANDS,
    )

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Audio Probe: {audio_path.name}")
    print(f"{'='*60}\n")

    # Load
    y, sr = load_audio(audio_path, sr=args.sr)
    duration = len(y) / sr
    print(f"  Duration:    {duration:.2f}s")
    print(f"  Sample Rate: {sr}")
    print(f"  Samples:     {len(y)}")

    # BPM
    bpm_result = compute_bpm(
        y, sr,
        mode="auto_then_override",
        manual_bpm=args.bpm,
        doubletime_hint=True,
    )
    print(f"\n  BPM:")
    print(f"    Detected:  {bpm_result.get('raw_detected_bpm', 'N/A')}")
    print(f"    Used:      {bpm_result['bpm']:.1f}")
    print(f"    Source:    {bpm_result['source']}")
    print(f"    Confidence:{bpm_result['bpm_confidence']:.2f}")

    # Beat grid
    grid = compute_beat_grid(bpm_result["bpm"], sr, duration)
    print(f"\n  Beat Grid:")
    print(f"    Bars:      {grid['total_bars']}")
    print(f"    Beats:     {len(grid['beat_times'])}")
    print(f"    First 10 beat times: {[f'{t:.3f}' for t in grid['beat_times'][:10]]}")

    # Features
    features = compute_features(y, sr, bands=DEFAULT_BANDS)
    print(f"\n  Band Energies (average):")
    for band, data in features.get("band_rms", {}).items():
        if data:
            avg = np.mean(data)
            peak = np.max(data)
            print(f"    {band:8s}: avg={avg:.4f}  peak={peak:.4f}")

    print(f"\n  Onsets:      {features.get('onset_count', 0)}")
    onset_times = features.get("onset_times", [])
    if onset_times:
        print(f"  First 10:    {[f'{t:.3f}' for t in onset_times[:10]]}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
