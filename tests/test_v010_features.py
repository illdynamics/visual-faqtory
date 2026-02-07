#!/usr/bin/env python3
"""
test_v010_features.py - Tests for v0.1.0-alpha features
═══════════════════════════════════════════════════════════════════════════════

Tests:
  - Deterministic Prompt Synthesis
  - Evolution Line Loading
  - Base Folder Selection
  - Audio Analysis (mock-safe)
  - Motion Bucket ID Mapping
  - Cycle Timing
  - Video Preprocessing (ffmpeg-safe)

Run: python -m pytest tests/test_v010_features.py -v
  or: python tests/test_v010_features.py

Part of QonQrete Visual FaQtory v0.1.0-alpha
"""
import sys
import tempfile
from pathlib import Path

# Ensure vfaq is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_synthesize_prompt_deterministic():
    """Same cycle_index + same inputs = identical output."""
    from vfaq.prompt_synth import synthesize_prompt

    lines = ["mutation A", "mutation B", "mutation C", "mutation D"]
    result1 = synthesize_prompt("base prompt", "style hint", lines, cycle_index=3)
    result2 = synthesize_prompt("base prompt", "style hint", lines, cycle_index=3)
    assert result1 == result2, "Must be deterministic"

    result3 = synthesize_prompt("base prompt", "style hint", lines, cycle_index=4)
    assert result1 != result3, "Different cycles should produce different prompts"


def test_synthesize_prompt_structure():
    """Prompt must have base, style hints, and mutations."""
    from vfaq.prompt_synth import synthesize_prompt

    lines = ["mutation X", "mutation Y"]
    result = synthesize_prompt("my base", "my style", lines, cycle_index=0)

    assert "my base" in result
    assert "STYLE_HINTS: my style" in result
    assert "EVOLUTION_MUTATIONS:" in result


def test_synthesize_prompt_empty_style():
    """Empty style hints should not add STYLE_HINTS section."""
    from vfaq.prompt_synth import synthesize_prompt

    result = synthesize_prompt("base", "", [], cycle_index=0)
    assert "STYLE_HINTS:" not in result


def test_synthesize_video_prompt_has_motion():
    """Video prompt should include motion at the end."""
    from vfaq.prompt_synth import synthesize_video_prompt

    result = synthesize_video_prompt(
        "base", "style", "slow dolly push", [], cycle_index=0
    )
    assert "MOTION: slow dolly push" in result


def test_evolution_line_selection_deterministic():
    """Selection must be deterministic and avoid duplicates."""
    from vfaq.prompt_synth import select_evolution_mutations

    lines = [f"line_{i}" for i in range(20)]
    sel1 = select_evolution_mutations(lines, cycle_index=5, max_mutations=3)
    sel2 = select_evolution_mutations(lines, cycle_index=5, max_mutations=3)
    assert sel1 == sel2

    # Check no duplicates
    assert len(sel1) == len(set(sel1))


def test_load_evolution_lines_default():
    """Should return defaults when file is missing."""
    from vfaq.prompt_synth import load_evolution_lines, DEFAULT_EVOLUTION_LINES

    with tempfile.TemporaryDirectory() as tmpdir:
        lines = load_evolution_lines(Path(tmpdir))
        assert len(lines) == len(DEFAULT_EVOLUTION_LINES)


def test_load_evolution_lines_from_file():
    """Should load lines from file."""
    from vfaq.prompt_synth import load_evolution_lines

    with tempfile.TemporaryDirectory() as tmpdir:
        evo_file = Path(tmpdir) / "evolution_lines.md"
        evo_file.write_text("# Header\n- line one\n- line two\n\n- line three\n")
        lines = load_evolution_lines(Path(tmpdir))
        assert lines == ["line one", "line two", "line three"]


def test_motion_bucket_mapping():
    """Motion keywords should map to bucket IDs."""
    from vfaq.prompt_synth import map_motion_to_bucket_id

    assert map_motion_to_bucket_id("slow cinematic push forward") != 127
    assert map_motion_to_bucket_id("chaotic glitch storm") >= 150
    assert map_motion_to_bucket_id("") == 127  # default
    assert map_motion_to_bucket_id("something unknown") == 127


def test_base_folder_selection_newest():
    """Should pick newest file."""
    import os
    import time
    from vfaq.base_folders import select_base_files

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "base_image"
        img_dir.mkdir()
        old = img_dir / "old.png"
        old.write_bytes(b"old")
        time.sleep(0.1)
        new = img_dir / "new.png"
        new.write_bytes(b"new")

        config = {"inputs": {"base_folders": {
            "enabled": True, "pick_mode": "newest",
            "base_image_dir": "base_image",
            "base_audio_dir": "base_audio",
            "base_video_dir": "base_video",
            "allow_empty": True,
        }}}
        result = select_base_files(Path(tmpdir), config)
        assert result["base_image"].name == "new.png"


def test_base_folder_selection_random_deterministic():
    """Random mode should be deterministic with same seed."""
    from vfaq.base_folders import select_base_files

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "base_image"
        img_dir.mkdir()
        for i in range(5):
            (img_dir / f"img_{i}.png").write_bytes(b"x")

        config = {"inputs": {"base_folders": {
            "enabled": True, "pick_mode": "random", "random_seed": 42,
            "base_image_dir": "base_image",
            "base_audio_dir": "base_audio",
            "base_video_dir": "base_video",
            "allow_empty": True,
        }}}
        r1 = select_base_files(Path(tmpdir), config)
        r2 = select_base_files(Path(tmpdir), config)
        assert r1["base_image"].name == r2["base_image"].name


def test_cycle_timing_bpm_sync():
    """Cycle durations should be BPM-synced."""
    from vfaq.audio_reactivity import compute_cycle_timing

    timing = compute_cycle_timing(bpm=174.0, bars_per_cycle=8, total_cycles=3)
    # 174 bpm → beat = 60/174 ≈ 0.345s → bar = 1.379s → 8 bars ≈ 11.03s
    assert abs(timing["cycle_duration_sec"] - 11.034) < 0.1
    assert len(timing["cycles"]) == 3
    assert timing["cycles"][0]["start_time"] == 0.0


def test_beat_grid_generation():
    """Beat grid should have correct structure."""
    from vfaq.audio_reactivity import compute_beat_grid

    grid = compute_beat_grid(bpm=120.0, sr=44100, duration=10.0)
    assert grid["bpm"] == 120.0
    assert grid["total_bars"] > 0
    assert len(grid["beat_times"]) > 0
    assert len(grid["bar_times"]) > 0
    # At 120 bpm, beat = 0.5s, so 10s should have ~20 beats
    assert abs(len(grid["beat_times"]) - 20) <= 1


def test_video_preprocess_missing_file():
    """Should raise FileNotFoundError for missing input."""
    from vfaq.video_preprocess import preprocess_video

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            preprocess_video(
                Path(tmpdir) / "nonexistent.mp4",
                Path(tmpdir) / "out.mp4"
            )
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


def test_briq_new_fields_backward_compat():
    """Old briq JSONs without new fields should load cleanly."""
    from vfaq.visual_briq import VisualBriq

    old_data = {
        "briq_id": "test_0001_abc",
        "cycle_index": 0,
        "created_at": "2026-01-01T00:00:00",
        "mode": "text",
        "prompt": "test",
        "negative_prompt": "bad",
        "style_tags": [],
        "quality_tags": [],
        "seed": 42,
        "base_image_path": None,
        "base_video_path": None,
        "spec": {"width": 1024, "height": 576, "cfg_scale": 7.0, "steps": 30,
                 "sampler": "euler", "video_frames": 25, "video_fps": 8,
                 "clip_seconds": 8.0, "motion_bucket_id": 127,
                 "noise_aug_strength": 0.02, "denoise_strength": 0.4},
        "status": "pending",
        "raw_video_path": None,
        "looped_video_path": None,
        "source_image_path": None,
        "generation_time": 0.0,
        "backend_used": "",
        "error_message": None,
        "evolution_suggestion": None,
        "suggested_prompt_delta": None,
    }
    briq = VisualBriq.from_dict(old_data)
    assert briq.evolution_mutations == []
    assert briq.audio_segment_stats == {}
    assert briq.bpm == 0.0
    assert briq.v2v_preprocessed_path is None


if __name__ == "__main__":
    tests = [
        test_synthesize_prompt_deterministic,
        test_synthesize_prompt_structure,
        test_synthesize_prompt_empty_style,
        test_synthesize_video_prompt_has_motion,
        test_evolution_line_selection_deterministic,
        test_load_evolution_lines_default,
        test_load_evolution_lines_from_file,
        test_motion_bucket_mapping,
        test_base_folder_selection_newest,
        test_base_folder_selection_random_deterministic,
        test_cycle_timing_bpm_sync,
        test_beat_grid_generation,
        test_video_preprocess_missing_file,
        test_briq_new_fields_backward_compat,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)
