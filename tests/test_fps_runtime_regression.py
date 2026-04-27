import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from vfaq.run_state import RunState
from vfaq.visual_faqtory import VisualFaQtory


_MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
    0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
    0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
    0x44, 0xAE, 0x42, 0x60, 0x82,
])


class FpsRuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="vfaq_fps_regression_"))
        self.worqspace_dir = self.temp_dir / "worqspace"
        self.worqspace_dir.mkdir(parents=True, exist_ok=True)
        (self.worqspace_dir / "story.txt").write_text("Para one.\n\nPara two.", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _base_config(self, mode: str = "text", video_fps: float = 12.0):
        return {
            "backend": {"type": "hybrid", "width": 320, "height": 180, "mock_delay": 0.0},
            "image_backend": {"type": "mock"},
            "video_backend": {"type": "mock"},
            "morph_backend": {"type": "mock"},
            "input": {"mode": mode},
            "paragraph_story": {
                "max_paragraphs": 2,
                "img2vid_duration_sec": 0.5,
                "video_fps": video_fps,
                "timing_authority": "duration",
                "enable_loop_closure": False,
            },
            "finalizer": {"enabled": False, "per_cycle_interpolation": False},
        }

    def _write_config(self, config: dict):
        (self.worqspace_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _seed_engine_outputs(qodeyard_dir: Path, cycle_idx: int, checkpoint_callback):
        videos_dir = qodeyard_dir / "videos"
        frames_dir = qodeyard_dir / "frames"
        videos_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / f"video_{cycle_idx:03d}.mp4"
        last_frame_path = frames_dir / f"lastframe_{cycle_idx:03d}.png"
        anchor_path = frames_dir / "anchor_frame_001.png"
        video_path.write_bytes(f"video-{cycle_idx}".encode("utf-8"))
        last_frame_path.write_bytes(_MINIMAL_PNG)
        anchor_path.write_bytes(_MINIMAL_PNG)
        (qodeyard_dir / "final_output.mp4").write_bytes(b"final-output")
        if checkpoint_callback:
            checkpoint_callback(cycle_idx, last_frame_path, video_path, anchor_path)

    def test_text_mode_reinject_enabled_propagates_video_fps(self):
        config = self._base_config(mode="text", video_fps=11.0)
        self._write_config(config)
        captured = {}

        def fake_run_sliding_story(*, qodeyard_dir, config, base_image_path, base_video_path, **engine_kwargs):
            captured["video_fps"] = config.video_fps
            captured["reinject"] = config.reinject
            captured["base_image_path"] = base_image_path
            captured["base_video_path"] = base_video_path
            self._seed_engine_outputs(qodeyard_dir, 1, engine_kwargs.get("checkpoint_callback"))
            return qodeyard_dir / "final_output.mp4"

        with patch("vfaq.visual_faqtory.run_sliding_story", new=fake_run_sliding_story):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=self.temp_dir / "run_text",
                dry_run=False,
                project_name="fps-text-reinject",
                reinject=True,
            )
            vf.run()

        self.assertEqual(captured["video_fps"], 11.0)
        self.assertTrue(captured["reinject"])
        self.assertIsNone(captured["base_image_path"])
        self.assertIsNone(captured["base_video_path"])

    def test_image_mode_uses_detected_base_image_path(self):
        config = self._base_config(mode="image", video_fps=9.0)
        self._write_config(config)
        base_images = self.worqspace_dir / "base_images"
        base_images.mkdir(parents=True, exist_ok=True)
        source_image = base_images / "source.png"
        source_image.write_bytes(_MINIMAL_PNG)
        captured = {}

        def fake_run_sliding_story(*, qodeyard_dir, base_image_path, base_video_path, **engine_kwargs):
            captured["base_image_path"] = base_image_path
            captured["base_video_path"] = base_video_path
            self._seed_engine_outputs(qodeyard_dir, 1, engine_kwargs.get("checkpoint_callback"))
            return qodeyard_dir / "final_output.mp4"

        with patch("vfaq.visual_faqtory.run_sliding_story", new=fake_run_sliding_story):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=self.temp_dir / "run_image",
                dry_run=False,
                project_name="fps-image-mode",
            )
            vf.run()

        self.assertEqual(Path(captured["base_image_path"]).resolve(), source_image.resolve())
        self.assertIsNone(captured["base_video_path"])

    def test_video_mode_uses_extracted_frame_for_non_veo_backends(self):
        config = self._base_config(mode="video", video_fps=7.0)
        self._write_config(config)
        base_video_dir = self.worqspace_dir / "base_video"
        base_video_dir.mkdir(parents=True, exist_ok=True)
        source_video = base_video_dir / "source.mp4"
        source_video.write_bytes(b"fake-video")
        captured = {}

        def fake_extract(video_path, output_path, width=1024, height=576):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(_MINIMAL_PNG)
            return True

        def fake_run_sliding_story(*, qodeyard_dir, base_image_path, base_video_path, **engine_kwargs):
            captured["base_image_path"] = base_image_path
            captured["base_video_path"] = base_video_path
            self._seed_engine_outputs(qodeyard_dir, 1, engine_kwargs.get("checkpoint_callback"))
            return qodeyard_dir / "final_output.mp4"

        with patch("vfaq.visual_faqtory._extract_video_frame", new=fake_extract), \
             patch("vfaq.visual_faqtory.run_sliding_story", new=fake_run_sliding_story):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=self.temp_dir / "run_video",
                dry_run=False,
                project_name="fps-video-mode",
            )
            vf.run()

        self.assertIsNotNone(captured["base_image_path"])
        self.assertEqual(captured["base_image_path"].name, "extracted_frame.png")
        self.assertIsNone(captured["base_video_path"])

    def test_resume_path_uses_frozen_meta_video_fps(self):
        current_cfg = self._base_config(mode="text", video_fps=99.0)
        self._write_config(current_cfg)

        run_dir = self.temp_dir / "run_resume"
        meta_dir = run_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        resume_cfg = self._base_config(mode="text", video_fps=3.0)
        (meta_dir / "config.yaml").write_text(yaml.safe_dump(resume_cfg, sort_keys=False), encoding="utf-8")
        (meta_dir / "story.txt").write_text("Para one.\n\nPara two.", encoding="utf-8")

        videos_dir = run_dir / "videos"
        frames_dir = run_dir / "frames"
        videos_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "video_001.mp4").write_bytes(b"video-1")
        (frames_dir / "lastframe_001.png").write_bytes(_MINIMAL_PNG)
        (frames_dir / "anchor_frame_001.png").write_bytes(_MINIMAL_PNG)

        state = RunState(
            run_id="resume-fps",
            status="running",
            backend_type="split(image=mock, video=mock, morph=mock)",
            cycles_planned=2,
            cycles_completed=1,
            next_cycle_index=2,
            last_completed_cycle=1,
            final_video_paths=[str(videos_dir / "video_001.mp4")],
            completed_cycle_indices=[1],
            resume_enabled=True,
        )
        (run_dir / "faqtory_state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

        captured = {}

        def fake_run_sliding_story(*, qodeyard_dir, config, **engine_kwargs):
            captured["video_fps"] = config.video_fps
            captured["start_cycle"] = engine_kwargs.get("start_cycle")
            self._seed_engine_outputs(qodeyard_dir, 2, engine_kwargs.get("checkpoint_callback"))
            return qodeyard_dir / "final_output.mp4"

        with patch("vfaq.visual_faqtory.run_sliding_story", new=fake_run_sliding_story):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=run_dir,
                dry_run=False,
                resume=True,
                project_name="fps-resume-mode",
            )
            vf.run()

        self.assertEqual(captured["video_fps"], 3.0)
        self.assertEqual(captured["start_cycle"], 2)


if __name__ == "__main__":
    unittest.main()
