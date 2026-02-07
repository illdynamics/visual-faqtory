#!/usr/bin/env python3
"""
test_v025_features.py - Tests for v0.2.5-beta features
═══════════════════════════════════════════════════════════════════════════════

Tests for:
  - Auto-Duration planning (v0.1.2)
  - Stream Engine context extraction (v0.2.0)
  - VisualBriq stream fields (v0.2.0)
  - Sanitization + moderation (v0.2.5)
  - Crowd queue (v0.2.5)
  - Overlay writer (v0.2.5)
  - TurboEngine prompt state (v0.2.5)
  - Backend generate_stream_video interface (v0.2.0)

Part of QonQrete Visual FaQtory v0.2.5-beta
"""
import os
import sys
import time
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vfaq.visual_briq import VisualBriq, GenerationSpec, InputMode, BriqStatus, generate_briq_id
from vfaq.duration_planner import (
    plan_duration, compute_cycle_duration, detect_audio_duration,
    trim_video, mux_audio_video, post_finalize_trim_and_mux,
)
from vfaq.stream_engine import (
    get_stream_config, prepare_stream_cycle,
    compute_beat_aligned_generation_length,
)
from vfaq.utils_sanitize import sanitize_prompt, sanitize_name
from vfaq.crowd_queue import PromptQueue, PromptItem, RateLimiter
from vfaq.overlay_writer import OverlayWriter
from vfaq.backends import MockBackend, GenerationRequest


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DURATION TESTS (v0.1.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDurationPlanner(unittest.TestCase):
    """Test auto-duration planning logic."""

    def test_default_no_override(self):
        """Default config should not override cycles."""
        plan = plan_duration(config={}, requested_cycles=10)
        self.assertEqual(plan['required_cycles'], 10)
        self.assertIsNone(plan['override_reason'])

    def test_match_audio_computes_cycles(self):
        """match_audio=true should compute required cycles from audio duration."""
        config = {'duration': {'mode': 'auto', 'match_audio': True, 'mux_audio': True}}
        # Mock: 120s audio, 174 BPM, 8 bars/cycle → cycle_dur = 8*4*(60/174) ≈ 11.03s
        # required = ceil(120/11.03) = 11
        fake_audio = MagicMock()
        fake_audio.exists.return_value = True
        with patch('vfaq.duration_planner.detect_audio_duration', return_value=120.0):
            plan = plan_duration(
                config=config,
                audio_path=fake_audio,
                bpm=174,
                bars_per_cycle=8,
                requested_cycles=5,
            )
        self.assertEqual(plan['required_cycles'], 11)
        self.assertIsNotNone(plan['override_reason'])
        self.assertAlmostEqual(plan['trim_to'], 120.0)

    def test_fixed_mode(self):
        """Fixed mode should compute cycles from fixed seconds."""
        config = {'duration': {'mode': 'fixed', 'seconds': 60}}
        plan = plan_duration(config=config, bpm=120, bars_per_cycle=4)
        # cycle_dur = 4 * 4 * (60/120) = 8s, required = ceil(60/8) = 8
        self.assertEqual(plan['required_cycles'], 8)
        self.assertEqual(plan['trim_to'], 60)

    def test_unlimited_mode_no_change(self):
        """Unlimited mode should not change behavior."""
        config = {'duration': {'mode': 'unlimited'}}
        plan = plan_duration(config=config, requested_cycles=20)
        self.assertEqual(plan['required_cycles'], 20)
        self.assertIsNone(plan['override_reason'])

    def test_compute_cycle_duration(self):
        """Cycle duration formula: bars × 4 × (60/BPM)."""
        dur = compute_cycle_duration(bpm=174, bars_per_cycle=8)
        expected = 8 * 4 * (60.0 / 174)
        self.assertAlmostEqual(dur, expected, places=4)

    def test_match_audio_disabled_no_override(self):
        """match_audio=false should not override even with audio present."""
        config = {'duration': {'mode': 'auto', 'match_audio': False}}
        fake_audio = MagicMock()
        fake_audio.exists.return_value = True
        with patch('vfaq.duration_planner.detect_audio_duration', return_value=120.0):
            plan = plan_duration(
                config=config,
                audio_path=fake_audio,
                bpm=174,
                requested_cycles=5,
            )
        self.assertEqual(plan['required_cycles'], 5)
        self.assertIsNone(plan['override_reason'])


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM ENGINE TESTS (v0.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamEngine(unittest.TestCase):
    """Test stream mode context extraction and beat alignment."""

    def test_get_stream_config_defaults(self):
        """Default stream config should be disabled."""
        cfg = get_stream_config({})
        self.assertFalse(cfg['enabled'])
        self.assertEqual(cfg['context_length'], 24)
        self.assertEqual(cfg['generation_length'], 72)

    def test_get_stream_config_enabled(self):
        """Enabled stream config reads correctly."""
        cfg = get_stream_config({
            'stream_mode': {'enabled': True, 'method': 'longcat', 'context_length': 48}
        })
        self.assertTrue(cfg['enabled'])
        self.assertEqual(cfg['method'], 'longcat')
        self.assertEqual(cfg['context_length'], 48)

    def test_beat_aligned_generation_no_bpm(self):
        """No BPM should return unchanged generation length."""
        result = compute_beat_aligned_generation_length(72, 8, 0)
        self.assertEqual(result, 72)

    def test_beat_aligned_generation_with_bpm(self):
        """Should snap to nearest beat boundary."""
        result = compute_beat_aligned_generation_length(72, 8, 174)
        # 72/8 = 9s generation, 60/174 = 0.345s/beat
        # 9/0.345 ≈ 26.1 beats → round to 26 → 26*0.345 = 8.97s → 8.97*8 ≈ 72 frames
        # Within ±10% so should be valid
        self.assertGreater(result, 0)
        self.assertAlmostEqual(result, 72, delta=8)

    def test_prepare_stream_cycle_zero(self):
        """Cycle 0 should return no context."""
        result = prepare_stream_cycle(
            cycle_index=0, previous_video=None,
            output_dir=Path('/tmp/test'), stream_config=get_stream_config({}),
        )
        self.assertIsNone(result['context_video_path'])


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL BRIQ STREAM FIELDS (v0.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVisualBriqStreamFields(unittest.TestCase):
    """Test new stream mode fields in VisualBriq."""

    def test_new_fields_default(self):
        """New stream fields should have correct defaults."""
        briq = VisualBriq(briq_id="test", cycle_index=0)
        self.assertIsNone(briq.context_video_path)
        self.assertIsNone(briq.flow_state)
        self.assertIsNone(briq.stream_video_path)

    def test_generation_spec_stream_fields(self):
        """GenerationSpec should have stream fields."""
        spec = GenerationSpec()
        self.assertEqual(spec.context_duration, 1.5)
        self.assertIsNone(spec.context_frames)
        self.assertIsNone(spec.generation_frames)
        self.assertEqual(spec.overlap_frames, 0)

    def test_serialization_roundtrip(self):
        """New fields should survive serialization."""
        briq = VisualBriq(briq_id="test_stream", cycle_index=5)
        briq.context_video_path = Path("/tmp/ctx.mp4")
        briq.stream_video_path = Path("/tmp/stream.mp4")
        briq.flow_state = {"velocity": [0.1, -0.2]}
        briq.spec.generation_frames = 72
        briq.spec.context_frames = 24

        d = briq.to_dict()
        restored = VisualBriq.from_dict(d)
        self.assertEqual(str(restored.context_video_path), "/tmp/ctx.mp4")
        self.assertEqual(str(restored.stream_video_path), "/tmp/stream.mp4")
        self.assertEqual(restored.flow_state, {"velocity": [0.1, -0.2]})
        self.assertEqual(restored.spec.generation_frames, 72)

    def test_backward_compat_old_briq(self):
        """Old briq JSON without stream fields should load fine."""
        old_dict = {
            'briq_id': 'old_briq', 'cycle_index': 0,
            'mode': 'text', 'status': 'complete',
            'created_at': '2025-01-01T00:00:00',
            'spec': {'width': 1024, 'height': 576, 'cfg_scale': 7.0,
                     'steps': 30, 'sampler': 'euler_ancestral',
                     'video_frames': 25, 'video_fps': 8, 'clip_seconds': 8.0,
                     'motion_bucket_id': 127, 'noise_aug_strength': 0.02,
                     'denoise_strength': 0.4},
            'prompt': 'test', 'negative_prompt': '',
            'style_tags': [], 'quality_tags': [],
            'seed': 42,
        }
        briq = VisualBriq.from_dict(old_dict)
        self.assertIsNone(briq.context_video_path)
        self.assertEqual(briq.spec.context_duration, 1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# SANITIZATION TESTS (v0.2.5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitization(unittest.TestCase):
    """Test crowd prompt sanitization."""

    def test_basic_sanitize(self):
        text, ok, reason = sanitize_prompt("neon skull dreamscape")
        self.assertTrue(ok)
        self.assertEqual(text, "neon skull dreamscape")

    def test_too_short(self):
        text, ok, reason = sanitize_prompt("hi")
        self.assertFalse(ok)
        self.assertIn("short", reason)

    def test_too_long_truncates(self):
        long_text = "a" * 200
        text, ok, reason = sanitize_prompt(long_text, {'max_len': 50, 'min_len': 1})
        self.assertTrue(ok)
        self.assertLessEqual(len(text), 50)

    def test_banned_word_blocked(self):
        text, ok, reason = sanitize_prompt("something nazi something")
        self.assertFalse(ok)
        self.assertIn("banned", reason)

    def test_url_stripped(self):
        text, ok, reason = sanitize_prompt("check https://evil.com for cool stuff")
        self.assertTrue(ok)
        self.assertNotIn("http", text)

    def test_whitespace_collapsed(self):
        text, ok, reason = sanitize_prompt("  neon   skull   vibes  ")
        self.assertTrue(ok)
        self.assertEqual(text, "neon skull vibes")

    def test_empty_prompt(self):
        text, ok, reason = sanitize_prompt("")
        self.assertFalse(ok)

    def test_sanitize_name(self):
        self.assertEqual(sanitize_name(""), "anon")
        self.assertEqual(sanitize_name("Dave"), "Dave")
        name = sanitize_name("A" * 100)
        self.assertLessEqual(len(name), 30)


# ═══════════════════════════════════════════════════════════════════════════════
# CROWD QUEUE TESTS (v0.2.5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrowdQueue(unittest.TestCase):
    """Test crowd prompt queue."""

    def setUp(self):
        self.queue = PromptQueue()

    def test_submit_and_pop(self):
        ok, reason = self.queue.submit("neon skull", "Dave", "1.2.3.4")
        self.assertTrue(ok)
        self.assertEqual(self.queue.depth(), 1)

        item = self.queue.pop_next()
        self.assertIsNotNone(item)
        self.assertEqual(item.prompt, "neon skull")
        self.assertEqual(item.name, "Dave")
        self.assertEqual(self.queue.depth(), 0)

    def test_peek_doesnt_remove(self):
        self.queue.submit("test prompt", "Bob", "1.2.3.4")
        item = self.queue.peek_next()
        self.assertIsNotNone(item)
        self.assertEqual(self.queue.depth(), 1)

    def test_pop_empty_returns_none(self):
        self.assertIsNone(self.queue.pop_next())

    def test_list_top(self):
        for i in range(5):
            self.queue.submit(f"prompt {i}", f"user{i}", f"10.0.0.{i}")
        items = self.queue.list_top(3)
        self.assertEqual(len(items), 3)

    def test_max_depth(self):
        q = PromptQueue({'queue': {'max_depth': 3}})
        for i in range(5):
            q.submit(f"prompt {i}", f"user{i}", f"10.0.0.{i}")
        self.assertLessEqual(q.depth(), 3)

    def test_stats(self):
        self.queue.submit("test", "X", "1.1.1.1")
        stats = self.queue.stats()
        self.assertEqual(stats['depth'], 1)
        self.assertEqual(stats['accepted'], 1)

    def test_moderation_blocks_banned(self):
        q = PromptQueue({'moderation': {'banned_words': ['spam']}})
        ok, reason = q.submit("this is spam content", "Bot", "1.1.1.1")
        self.assertFalse(ok)


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER TESTS (v0.2.5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimiter(unittest.TestCase):
    """Test rate limiting."""

    def test_allows_within_limit(self):
        rl = RateLimiter(window_seconds=10, max_requests=3)
        ok1, _ = rl.check("user1")
        ok2, _ = rl.check("user1")
        ok3, _ = rl.check("user1")
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertTrue(ok3)

    def test_blocks_over_limit(self):
        rl = RateLimiter(window_seconds=10, max_requests=2)
        rl.check("user1")
        rl.check("user1")
        ok, reason = rl.check("user1")
        self.assertFalse(ok)
        self.assertIn("rate limited", reason)

    def test_different_keys_independent(self):
        rl = RateLimiter(window_seconds=10, max_requests=1)
        ok1, _ = rl.check("user1")
        ok2, _ = rl.check("user2")
        self.assertTrue(ok1)
        self.assertTrue(ok2)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAY WRITER TESTS (v0.2.5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlayWriter(unittest.TestCase):
    """Test OBS overlay file writer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = OverlayWriter(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_now(self):
        self.writer.write_now("test prompt", "DROP", "active crowd")
        content = (Path(self.tmpdir) / "now.txt").read_text()
        self.assertIn("test prompt", content)

    def test_write_next(self):
        self.writer.write_next("@Dave: neon skull")
        content = (Path(self.tmpdir) / "next.txt").read_text()
        self.assertIn("Dave", content)

    def test_write_queue(self):
        items = [
            {'name': 'Alice', 'prompt': 'fire'},
            {'name': 'Bob', 'prompt': 'water'},
        ]
        self.writer.write_queue(items)
        content = (Path(self.tmpdir) / "queue.txt").read_text()
        self.assertIn("Alice", content)
        self.assertIn("Bob", content)

    def test_write_toast(self):
        self.writer.write_toast("Dave", "neon skull")
        content = (Path(self.tmpdir) / "toast.txt").read_text()
        self.assertIn("Dave", content)
        self.assertIn("neon skull", content)

    def test_atomic_write(self):
        """Verify no .tmp files remain."""
        self.writer.write_now("test")
        tmp_files = list(Path(self.tmpdir).glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND STREAM INTERFACE TESTS (v0.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendStreamInterface(unittest.TestCase):
    """Test that backends support stream generation interface."""

    def test_mock_backend_has_stream(self):
        backend = MockBackend({})
        self.assertTrue(hasattr(backend, 'generate_stream_video'))

    def test_mock_stream_produces_output(self):
        backend = MockBackend({})
        tmpdir = Path(tempfile.mkdtemp())

        # Create a fake source video
        src = tmpdir / "source.mp4"
        src.write_bytes(b'\x00' * 100)

        request = GenerationRequest(
            prompt="test",
            seed=42,
            base_video_path=src,
            output_dir=tmpdir,
            atom_id="test_stream",
        )
        result = backend.generate_stream_video(request)
        self.assertTrue(result.success)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TURBO ENGINE PROMPT STATE TESTS (v0.2.5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTurboPromptState(unittest.TestCase):
    """Test TurboEngine prompt state management."""

    def test_prompt_state_defaults(self):
        from vfaq.turbo_engine import PromptState
        state = PromptState()
        self.assertEqual(state.macro, "CHILL")
        self.assertEqual(state.base_prompt, "")
        self.assertFalse(state.crowd_active)

    def test_crowd_active_expiry(self):
        from vfaq.turbo_engine import PromptState
        state = PromptState()
        state.crowd_prompt = "neon"
        state.crowd_expiry = time.time() + 100
        self.assertTrue(state.crowd_active)

        state.crowd_expiry = time.time() - 1
        self.assertFalse(state.crowd_active)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION: EXISTING MODES UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegressionExistingModes(unittest.TestCase):
    """Verify existing TEXT/IMAGE/VIDEO modes still work with MockBackend."""

    def test_text_mode_basic(self):
        from vfaq.construqtor import ConstruQtor
        tmpdir = Path(tempfile.mkdtemp())
        cfg = {'backend': {'type': 'mock'}, 'input': {'video2video': {'enabled': True}}}

        cq = ConstruQtor(config=cfg, qodeyard_dir=tmpdir)
        briq = VisualBriq(briq_id="reg_text", cycle_index=0, mode=InputMode.TEXT, prompt="test prompt", seed=42)
        result = cq.construct(briq)
        self.assertEqual(result.status, BriqStatus.CONSTRUCTED)
        self.assertIsNotNone(result.raw_video_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stream_disabled_no_change(self):
        """With stream_mode disabled, construct should behave exactly like v0.1.1."""
        from vfaq.construqtor import ConstruQtor
        tmpdir = Path(tempfile.mkdtemp())
        cfg = {
            'backend': {'type': 'mock'},
            'input': {'video2video': {'enabled': True}},
            'stream_mode': {'enabled': False},
        }
        cq = ConstruQtor(config=cfg, qodeyard_dir=tmpdir)
        self.assertFalse(cq.stream_enabled)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
