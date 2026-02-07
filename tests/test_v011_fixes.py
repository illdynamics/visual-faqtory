#!/usr/bin/env python3
"""
test_v011_fixes.py — v0.1.1-alpha Acceptance Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for all four fixes in the v0.1.1-alpha spec:
  1. generate_video2video() exists and works on all backends
  2. ConstruQtor VIDEO mode calls V2V — no image fallback
  3. Graph-based CLIP injection replaces heuristic
  4. motion_prompt warning logged when workflow lacks text conditioning

Run: python tests/test_v011_fixes.py
"""
import os
import sys
import json
import shutil
import logging
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vfaq.backends import (
    MockBackend, ComfyUIBackend, GenerationRequest, GenerationResult,
    InputMode, FatalConfigError, SplitBackend
)
from vfaq.construqtor import ConstruQtor
from vfaq.visual_briq import VisualBriq, BriqStatus, InputMode as BriqInputMode


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: generate_video2video() exists and works
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateVideo2Video(unittest.TestCase):
    """Fix 1: Backend API for Video2Video."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock = MockBackend({'mock_delay': 0.0})

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mock_v2v_exists(self):
        """generate_video2video method exists on MockBackend."""
        self.assertTrue(hasattr(self.mock, 'generate_video2video'))
        self.assertTrue(callable(self.mock.generate_video2video))

    def test_mock_v2v_denoise_guard(self):
        """Denoise > 0.5 raises FatalConfigError."""
        req = GenerationRequest(
            prompt="test",
            denoise_strength=0.6,  # ❌ too high
            base_video_path=Path("/tmp/fake.mp4"),
            output_dir=Path(self.tmpdir),
            atom_id="test001"
        )
        with self.assertRaises(FatalConfigError) as ctx:
            self.mock.generate_video2video(req)
        self.assertIn("≤ 0.5", str(ctx.exception))

    def test_mock_v2v_denoise_boundary(self):
        """Denoise exactly 0.5 does NOT raise."""
        # Create a minimal input video
        video_path = Path(self.tmpdir) / "input.mp4"
        self._make_placeholder_video(video_path)

        req = GenerationRequest(
            prompt="test",
            denoise_strength=0.5,  # ✅ boundary
            base_video_path=video_path,
            output_dir=Path(self.tmpdir),
            atom_id="test002"
        )
        result = self.mock.generate_video2video(req)
        self.assertTrue(result.success)

    def test_mock_v2v_missing_video_fails(self):
        """V2V with no base_video_path returns failure (no crash)."""
        req = GenerationRequest(
            prompt="test",
            denoise_strength=0.35,
            base_video_path=None,
            output_dir=Path(self.tmpdir),
            atom_id="test003"
        )
        result = self.mock.generate_video2video(req)
        self.assertFalse(result.success)
        self.assertIn("base_video_path", result.error)

    def test_mock_v2v_produces_output(self):
        """V2V with valid input produces an output video file."""
        video_path = Path(self.tmpdir) / "input.mp4"
        self._make_placeholder_video(video_path)

        req = GenerationRequest(
            prompt="test",
            denoise_strength=0.35,
            base_video_path=video_path,
            output_dir=Path(self.tmpdir),
            atom_id="test004"
        )
        result = self.mock.generate_video2video(req)
        self.assertTrue(result.success)
        self.assertTrue(result.video_path.exists())
        self.assertGreater(result.video_path.stat().st_size, 0)

    def test_comfyui_v2v_exists(self):
        """generate_video2video method exists on ComfyUIBackend."""
        comfyui = ComfyUIBackend({})
        self.assertTrue(hasattr(comfyui, 'generate_video2video'))

    def test_split_backend_v2v_delegates(self):
        """SplitBackend delegates generate_video2video to video_backend."""
        mock_img = MockBackend({'mock_delay': 0.0})
        mock_vid = MockBackend({'mock_delay': 0.0})
        split = SplitBackend(mock_img, mock_vid)

        video_path = Path(self.tmpdir) / "input.mp4"
        self._make_placeholder_video(video_path)

        req = GenerationRequest(
            prompt="test",
            denoise_strength=0.35,
            base_video_path=video_path,
            output_dir=Path(self.tmpdir),
            atom_id="test005"
        )
        result = split.generate_video2video(req)
        self.assertTrue(result.success)

    def _make_placeholder_video(self, path: Path):
        """Create a minimal mp4 for testing."""
        try:
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', 'color=c=blue:s=64x64:d=1:r=8',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                str(path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg unavailable, write a fake file
            path.write_bytes(b'\x00' * 1024)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: ConstruQtor VIDEO mode uses V2V, no image fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstruQtorVideoMode(unittest.TestCase):
    """Fix 2: VIDEO mode must call generate_video2video, never image pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock = MockBackend({'mock_delay': 0.0})

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_video_mode_calls_v2v(self):
        """VIDEO mode calls generate_video2video, not generate_image."""
        video_path = Path(self.tmpdir) / "input.mp4"
        self._make_placeholder_video(video_path)

        config = {
            'input': {
                'video2video': {
                    'enabled': True,
                    'preprocess': {'width': 64, 'height': 64, 'fps': 8, 'duration_sec': 1},
                    'comfyui': {'sampler': {'denoise': 0.35, 'steps': 5, 'cfg': 4.5}}
                }
            }
        }
        cq = ConstruQtor(
            config=config,
            qodeyard_dir=Path(self.tmpdir) / "qodeyard",
            backend=self.mock
        )

        briq = VisualBriq(
            briq_id="v2v_test_001",
            cycle_index=0,
            mode=BriqInputMode.VIDEO,
            prompt="test v2v",
            negative_prompt="bad",
            base_video_path=video_path,
            seed=42,
        )

        # Track which backend methods are called
        original_gen_image = self.mock.generate_image
        original_gen_video = self.mock.generate_video
        calls = {'generate_image': 0, 'generate_video': 0, 'generate_video2video': 0}

        def track_image(req):
            calls['generate_image'] += 1
            return original_gen_image(req)
        def track_video(req, src):
            calls['generate_video'] += 1
            return original_gen_video(req, src)
        def track_v2v(req):
            calls['generate_video2video'] += 1
            return self.mock.generate_video2video.__wrapped__(self.mock, req) if hasattr(self.mock.generate_video2video, '__wrapped__') else MockBackend.generate_video2video(self.mock, req)

        self.mock.generate_image = track_image
        self.mock.generate_video = track_video
        # For V2V we need the real method
        orig_v2v = MockBackend.generate_video2video

        def patched_v2v(backend_self, req):
            calls['generate_video2video'] += 1
            return orig_v2v(backend_self, req)

        self.mock.generate_video2video = lambda req: patched_v2v(self.mock, req)

        result = cq.construct(briq)

        self.assertEqual(calls['generate_image'], 0, "generate_image should NOT be called in VIDEO mode")
        self.assertEqual(calls['generate_video'], 0, "generate_video should NOT be called in VIDEO mode")
        self.assertEqual(calls['generate_video2video'], 1, "generate_video2video MUST be called exactly once")
        self.assertEqual(result.status, BriqStatus.CONSTRUCTED)

    def test_video_mode_v2v_disabled_fails(self):
        """VIDEO mode with v2v disabled raises RuntimeError (no silent fallback)."""
        video_path = Path(self.tmpdir) / "input.mp4"
        self._make_placeholder_video(video_path)

        config = {
            'input': {
                'video2video': {'enabled': False}
            }
        }
        cq = ConstruQtor(
            config=config,
            qodeyard_dir=Path(self.tmpdir) / "qodeyard",
            backend=self.mock
        )

        briq = VisualBriq(
            briq_id="v2v_disabled_001",
            cycle_index=0,
            mode=BriqInputMode.VIDEO,
            prompt="test",
            base_video_path=video_path,
            seed=42,
        )

        with self.assertRaises(RuntimeError) as ctx:
            cq.construct(briq)
        self.assertIn("video2video", str(ctx.exception).lower())

    def test_text_mode_still_works(self):
        """TEXT mode unchanged: txt2img → img2vid."""
        config = {'input': {'video2video': {'enabled': True}}}
        cq = ConstruQtor(
            config=config,
            qodeyard_dir=Path(self.tmpdir) / "qodeyard",
            backend=self.mock
        )

        briq = VisualBriq(
            briq_id="text_test_001",
            cycle_index=0,
            mode=BriqInputMode.TEXT,
            prompt="beautiful landscape",
            seed=42,
        )
        result = cq.construct(briq)
        self.assertEqual(result.status, BriqStatus.CONSTRUCTED)
        self.assertIsNotNone(result.raw_video_path)

    def _make_placeholder_video(self, path: Path):
        try:
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', 'color=c=blue:s=64x64:d=1:r=8',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                str(path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            path.write_bytes(b'\x00' * 1024)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: Graph-based CLIP injection
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphBasedCLIPInjection(unittest.TestCase):
    """Fix 3: Replace heuristic CLIP injection with graph-based resolution."""

    def test_resolve_safe_v2v_workflow(self):
        """Graph resolution correctly identifies CLIP nodes in safe_video2video.json."""
        workflow = {
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "POSITIVE_PLACEHOLDER", "clip": ["2", 1]}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "NEGATIVE_PLACEHOLDER", "clip": ["2", 1]}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "model": ["2", 0],
                    "seed": 42,
                    "denoise": 0.35
                }
            }
        }

        resolved = ComfyUIBackend._resolve_clip_nodes_from_graph(workflow)
        self.assertEqual(resolved["positive"], ["3"])
        self.assertEqual(resolved["negative"], ["4"])

    def test_resolve_default_image_workflow(self):
        """Graph resolution works on the default SDXL image workflow."""
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "seed": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "masterpiece", "clip": ["4", 1]}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "low quality", "clip": ["4", 1]}
            }
        }

        resolved = ComfyUIBackend._resolve_clip_nodes_from_graph(workflow)
        self.assertEqual(resolved["positive"], ["6"])
        self.assertEqual(resolved["negative"], ["7"])

    def test_resolve_svd_workflow_no_clip(self):
        """SVD workflow (no CLIPTextEncode) returns empty lists."""
        workflow = {
            "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {}},
            "3": {"class_type": "SVD_img2vid_Conditioning", "inputs": {}},
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "model": ["1", 0],
                    "seed": 1
                }
            }
        }

        resolved = ComfyUIBackend._resolve_clip_nodes_from_graph(workflow)
        self.assertEqual(resolved["positive"], [])
        self.assertEqual(resolved["negative"], [])

    def test_inject_prompts_modifies_correct_nodes(self):
        """_inject_prompts_graph_based writes to the correct nodes only."""
        backend = ComfyUIBackend({})
        workflow = {
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "OLD_POS", "clip": ["2", 1]}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "OLD_NEG", "clip": ["2", 1]}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "seed": 42
                }
            }
        }

        result = backend._inject_prompts_graph_based(workflow, "NEW_POS", "NEW_NEG")
        self.assertEqual(result["3"]["inputs"]["text"], "NEW_POS")
        self.assertEqual(result["4"]["inputs"]["text"], "NEW_NEG")

    def test_heuristic_bypassed(self):
        """
        Regression: old heuristic would inject into nodes whose ID contains 'positive'.
        Graph-based should NOT do this; it follows KSampler references only.
        """
        backend = ComfyUIBackend({})
        workflow = {
            "positive_clip": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "UNRELATED", "clip": ["2", 1]}
            },
            "actual_positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "REAL_POS", "clip": ["2", 1]}
            },
            "actual_negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "REAL_NEG", "clip": ["2", 1]}
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["actual_positive", 0],
                    "negative": ["actual_negative", 0],
                    "seed": 1
                }
            }
        }

        result = backend._inject_prompts_graph_based(workflow, "INJECTED", "NEG_INJECTED")
        # positive_clip should NOT be touched (old heuristic would have touched it)
        self.assertEqual(result["positive_clip"]["inputs"]["text"], "UNRELATED")
        # actual_positive and actual_negative should be injected
        self.assertEqual(result["actual_positive"]["inputs"]["text"], "INJECTED")
        self.assertEqual(result["actual_negative"]["inputs"]["text"], "NEG_INJECTED")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: Motion prompt warning
# ═══════════════════════════════════════════════════════════════════════════════

class TestMotionPromptWarning(unittest.TestCase):
    """Fix 4: motion_prompt warning when workflow lacks text conditioning."""

    def test_warning_logged_for_svd_workflow(self):
        """Warning logged when motion_prompt set + no CLIPTextEncode in workflow."""
        backend = ComfyUIBackend({})
        workflow = {
            "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {}},
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "model": ["1", 0]
                }
            },
            "3": {"class_type": "SVD_img2vid_Conditioning", "inputs": {}}
        }
        request = GenerationRequest(
            prompt="test",
            motion_prompt="camera panning left slowly"
        )

        with self.assertLogs('vfaq.backends', level='WARNING') as cm:
            backend._warn_motion_prompt_if_ignored(workflow, request)

        found = any(
            "motion_prompt provided but ignored" in msg for msg in cm.output
        )
        self.assertTrue(found, f"Expected motion_prompt warning, got: {cm.output}")

    def test_no_warning_when_clip_present(self):
        """No warning when workflow HAS CLIPTextEncode nodes wired to KSampler."""
        backend = ComfyUIBackend({})
        workflow = {
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "pos", "clip": ["2", 1]}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "neg", "clip": ["2", 1]}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {"positive": ["3", 0], "negative": ["4", 0]}
            }
        }
        request = GenerationRequest(
            prompt="test",
            motion_prompt="camera panning"
        )

        # Should NOT produce a warning about motion_prompt being ignored
        import io
        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        test_logger = logging.getLogger('vfaq.backends')
        test_logger.addHandler(handler)
        try:
            backend._warn_motion_prompt_if_ignored(workflow, request)
            output = handler.stream.getvalue()
            self.assertNotIn("motion_prompt provided but ignored", output)
        finally:
            test_logger.removeHandler(handler)

    def test_no_warning_when_no_motion_prompt(self):
        """No warning when motion_prompt is empty/None."""
        backend = ComfyUIBackend({})
        workflow = {
            "4": {
                "class_type": "KSampler",
                "inputs": {"positive": ["3", 0], "negative": ["3", 1]}
            }
        }
        request = GenerationRequest(prompt="test", motion_prompt=None)

        # This should not log any warning (nothing to check)
        # Since no warning logged, assertLogs would fail — use a different approach
        import io
        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('vfaq.backends')
        logger.addHandler(handler)
        try:
            backend._warn_motion_prompt_if_ignored(workflow, request)
            output = handler.stream.getvalue()
            self.assertNotIn("motion_prompt", output)
        finally:
            logger.removeHandler(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION: Existing modes still work
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegressionExistingModes(unittest.TestCase):
    """Ensure TEXT and IMAGE modes are unaffected by v0.1.1 changes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock = MockBackend({'mock_delay': 0.0})

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_text_mode_unchanged(self):
        """TEXT mode: txt2img → img2vid → save."""
        config = {'input': {'video2video': {'enabled': True}}}
        cq = ConstruQtor(
            config=config,
            qodeyard_dir=Path(self.tmpdir) / "qodeyard",
            backend=self.mock
        )
        briq = VisualBriq(
            briq_id="txt_001", cycle_index=0,
            mode=BriqInputMode.TEXT, prompt="landscape", seed=42
        )
        result = cq.construct(briq)
        self.assertEqual(result.status, BriqStatus.CONSTRUCTED)
        self.assertIsNotNone(result.raw_video_path)

    def test_image_mode_unchanged(self):
        """IMAGE mode: base_image → img2vid → save."""
        img_path = Path(self.tmpdir) / "base.png"
        self._make_placeholder_image(img_path)

        config = {'input': {'video2video': {'enabled': True}}}
        cq = ConstruQtor(
            config=config,
            qodeyard_dir=Path(self.tmpdir) / "qodeyard",
            backend=self.mock
        )
        briq = VisualBriq(
            briq_id="img_001", cycle_index=0,
            mode=BriqInputMode.IMAGE, prompt="enhance",
            base_image_path=img_path, seed=42
        )
        result = cq.construct(briq)
        self.assertEqual(result.status, BriqStatus.CONSTRUCTED)
        self.assertIsNotNone(result.raw_video_path)

    def _make_placeholder_image(self, path: Path):
        """Create a minimal PNG."""
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        path.write_bytes(png_data)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Enable logging for tests that check log output
    logging.basicConfig(level=logging.DEBUG)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestGenerateVideo2Video,
        TestConstruQtorVideoMode,
        TestGraphBasedCLIPInjection,
        TestMotionPromptWarning,
        TestRegressionExistingModes,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # Pretty summary
    print("\n" + "=" * 50)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    for test, _ in result.failures:
        print(f"  ❌ {test}")
    for test, _ in result.errors:
        print(f"  💥 {test}")
    for test in [t for t, _ in []] or []:
        pass

    # Print passed tests
    for cls in [TestGenerateVideo2Video, TestConstruQtorVideoMode,
                TestGraphBasedCLIPInjection, TestMotionPromptWarning,
                TestRegressionExistingModes]:
        for method_name in loader.getTestCaseNames(cls):
            test_id = f"{cls.__name__}.{method_name}"
            failed_ids = [str(t) for t, _ in result.failures + result.errors]
            if not any(test_id in fid for fid in failed_ids):
                print(f"  ✅ {method_name}")

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {len(result.failures)} failed, {len(result.errors)} errors")
    print(f"{'=' * 50}")

    sys.exit(0 if result.wasSuccessful() else 1)
