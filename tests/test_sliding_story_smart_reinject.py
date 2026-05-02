import json
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vfaq.backends import GenerationResult
from vfaq.sliding_story_engine import (
    SlidingStoryConfig,
    run_sliding_story,
    shutdown_active_smart_reinject_workers,
)


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


class _FakeFinalizer:
    def __init__(self, project_dir, finalizer_config):
        self.project_dir = Path(project_dir)
        self.final_output_path = self.project_dir / "final_output.mp4"
        self.final_deliverable_path = self.project_dir / "final_60fps_1080p.mp4"

    def _process_cycle_video(self, video_path, resolved_fps):
        return None

    def finalize(self, cycle_video_paths):
        self.final_output_path.write_bytes(b"final")
        return self.final_output_path

    def run_post_stitch_finalizer(self):
        return None


class _RecordingBackend:
    def __init__(self, name: str, image_delay: float = 0.0):
        self.name = name
        self.image_delay = image_delay
        self.events = []
        self.video_call_times = []
        self.image_start_times = []
        self.image_end_times = []
        self.video_calls = []
        self.morph_calls = []
        self.image_calls = []
        self.video_requests = []
        self.morph_requests = []
        self._lock = threading.Lock()

    def _record(self, kind: str, atom_id: str, **kwargs):
        with self._lock:
            self.events.append({"kind": kind, "atom_id": atom_id, **kwargs})

    def _image_path(self, output_dir: Path, atom_id: str) -> Path:
        path = output_dir / f"{atom_id}_{self.name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_MINIMAL_PNG)
        return path

    def _video_path(self, output_dir: Path, atom_id: str) -> Path:
        path = output_dir / f"{atom_id}_{self.name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x00\x00\x20ftypisom" + atom_id.encode("utf-8"))
        return path

    def generate_image(self, request):
        start = time.time()
        self.image_start_times.append(start)
        self.image_calls.append(request.atom_id)
        self._record(
            "generate_image",
            request.atom_id,
            init_image=str(request.init_image_path) if request.init_image_path else None,
            prompt=request.prompt[:120],
            denoise=float(getattr(request, "denoise_strength", 0.0) or 0.0),
        )
        if self.image_delay > 0:
            time.sleep(self.image_delay)
        image_path = self._image_path(request.output_dir, request.atom_id)
        self.image_end_times.append(time.time())
        return GenerationResult(
            success=True,
            image_path=image_path,
            metadata={"backend": self.name, "operation": "image"},
        )

    def generate_video(self, request, source_image):
        self.video_call_times.append(time.time())
        source_str = str(source_image) if source_image else None
        self.video_calls.append((request.atom_id, source_str))
        self.video_requests.append(
            {
                "atom_id": request.atom_id,
                "source": source_str,
                "prompt": request.prompt,
                "video_prompt": request.video_prompt,
            }
        )
        self._record(
            "generate_video",
            request.atom_id,
            source=source_str,
            prompt=request.prompt[:160] if request.prompt else "",
            video_prompt=request.video_prompt[:160] if request.video_prompt else "",
        )
        video_path = self._video_path(request.output_dir, request.atom_id)
        return GenerationResult(
            success=True,
            video_path=video_path,
            metadata={"backend": self.name, "operation": "video"},
        )

    def generate_morph_video(self, request, start_image_path, end_image_path):
        self.morph_calls.append((request.atom_id, str(start_image_path), str(end_image_path)))
        self.morph_requests.append(
            {
                "atom_id": request.atom_id,
                "start_image": str(start_image_path),
                "end_image": str(end_image_path),
                "prompt": request.prompt,
                "video_prompt": request.video_prompt,
            }
        )
        self._record(
            "generate_morph_video",
            request.atom_id,
            start=str(start_image_path),
            end=str(end_image_path),
            prompt=request.prompt[:160] if request.prompt else "",
            video_prompt=request.video_prompt[:160] if request.video_prompt else "",
        )
        video_path = self._video_path(request.output_dir, request.atom_id)
        return GenerationResult(
            success=True,
            video_path=video_path,
            metadata={"backend": self.name, "operation": "morph"},
        )

    def check_availability(self):
        return True, "ok"


class _ReferenceStripThenI2VBackend(_RecordingBackend):
    """Simulates Venice stripping reference_image_urls then succeeding on i2v fallback."""

    def generate_video(self, request, source_image):
        if source_image is None and getattr(request, "reference_image_paths", None):
            self.video_call_times.append(time.time())
            self.video_calls.append((request.atom_id, None))
            self.video_requests.append(
                {
                    "atom_id": request.atom_id,
                    "source": None,
                    "prompt": request.prompt,
                    "video_prompt": request.video_prompt,
                }
            )
            self._record(
                "generate_video",
                request.atom_id,
                source=None,
                prompt=request.prompt[:160] if request.prompt else "",
                video_prompt=request.video_prompt[:160] if request.video_prompt else "",
            )
            video_path = self._video_path(request.output_dir, request.atom_id + "_ref_attempt")
            return GenerationResult(
                success=True,
                video_path=video_path,
                metadata={
                    "backend": self.name,
                    "operation": "video",
                    "response": {
                        "stripped_retry_fields": ["reference_image_urls"],
                        "accepted_optional_fields": [],
                    },
                },
            )
        return super().generate_video(request, source_image)


class _InterruptingBackend(_RecordingBackend):
    def __init__(self, name: str, interrupt_atom_id: str):
        super().__init__(name)
        self.interrupt_atom_id = interrupt_atom_id

    def generate_video(self, request, source_image):
        if request.atom_id == self.interrupt_atom_id:
            raise KeyboardInterrupt
        return super().generate_video(request, source_image)


class _FakeCrowdClient:
    def __init__(self, prompts):
        self._prompts = list(prompts)

    def pop_next(self):
        if not self._prompts:
            return None
        return self._prompts.pop(0)


class SmartReinjectSlidingStoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="vfaq_smart_reinject_"))
        self.story_path = self.temp_dir / "story.txt"
        self.story_path.write_text("Para one.\n\nPara two.\n\nPara three.", encoding="utf-8")
        self.base_image = self.temp_dir / "base.png"
        self.base_image.write_bytes(_MINIMAL_PNG)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_story_config(
        self,
        *,
        require_morph: bool,
        smart_enabled: bool = False,
        smart_use_morph: bool = True,
        smart_wait_timeout_sec: float = 0.0,
        smart_sync_fallback: bool = False,
        text_to_video_first_cycle: bool = True,
        enable_end_frame_morph: bool = True,
        crowd_enabled: bool = False,
        crowd_inject_mode: str = "append",
        crowd_inject_source_mode: str = "as_image_source",
        crowd_carryover_cycles: int = 0,
    ) -> SlidingStoryConfig:
        venice_cfg = {
            "text_to_video_first_cycle": text_to_video_first_cycle,
            "enable_end_frame_morph": enable_end_frame_morph,
            "video": {
                "text2vid": {"duration_seconds": 1},
                "img2vid": {"duration_seconds": 1},
                "aspect_ratio": "16:9",
                "resolution": "480p",
            },
        }
        backend_cfg = {
            "type": "hybrid",
            "image_backend": {"type": "venice"},
            "video_backend": {"type": "venice"},
            "morph_backend": {"type": "venice"},
            "venice": venice_cfg,
        }
        return SlidingStoryConfig(
            max_paragraphs=2,
            img2vid_duration_sec=1.0,
            img2img_denoise_min=0.3,
            img2img_denoise_max=0.5,
            require_morph=require_morph,
            seed_base=123,
            video_fps=4,
            timing_authority="duration",
            backend_config=backend_cfg,
            finalizer_config={"enabled": False, "per_cycle_interpolation": False, "per_cycle_pingpong": False},
            reinject=True,
            venice_config=venice_cfg,
            crowd_control_config=(
                {
                    "enabled": True,
                    "inject_mode": crowd_inject_mode,
                    "inject_source_mode": crowd_inject_source_mode,
                    "inject_label": "Audience mutation request",
                    "carryover_cycles": crowd_carryover_cycles,
                }
                if crowd_enabled
                else {"enabled": False}
            ),
            smart_reinject_enabled=smart_enabled,
            smart_reinject_every_n_cycles=1,
            smart_reinject_use_morph=smart_use_morph,
            smart_reinject_similarity_guard_enabled=True,
            smart_reinject_similarity_threshold=0.42,
            smart_reinject_wait_timeout_sec=smart_wait_timeout_sec,
            smart_reinject_sync_fallback=smart_sync_fallback,
            smart_reinject_denoise_min=0.25,
            smart_reinject_denoise_max=0.45,
            smart_reinject_prompt_prefix=(
                "Preserve the source image strongly. Make a subtle evolved keyframe for the next visual beat."
            ),
        )

    def _patch_runtime(self):
        def _fake_extract_last(video_path: Path, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(_MINIMAL_PNG)

        def _fake_extract_first(video_path: Path, output_path: Path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(_MINIMAL_PNG)

        return patch("vfaq.sliding_story_engine._extract_last_frame_ffmpeg", new=_fake_extract_last), \
            patch("vfaq.sliding_story_engine._extract_first_frame_ffmpeg", new=_fake_extract_first), \
            patch("vfaq.sliding_story_engine._resize_frame_to_target", new=lambda *args, **kwargs: None), \
            patch("vfaq.sliding_story_engine.Finalizer", new=_FakeFinalizer)

    def _run_story(
        self,
        *,
        run_name: str,
        config: SlidingStoryConfig,
        main_backend: _RecordingBackend,
        async_backend: _RecordingBackend = None,
        max_cycles: int = 2,
        similarity_score: float = 0.9,
        base_image: bool = False,
        crowd_prompts=None,
    ):
        run_dir = self.temp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        calls = {"count": 0}

        def _fake_create_backend(cfg):
            calls["count"] += 1
            if calls["count"] == 1:
                return main_backend
            if async_backend is not None and calls["count"] == 2:
                return async_backend
            return main_backend

        patchers = list(self._patch_runtime())
        patchers.append(patch("vfaq.sliding_story_engine.create_backend", new=_fake_create_backend))
        patchers.append(
            patch("vfaq.sliding_story_engine.calculate_frame_similarity", return_value=similarity_score)
        )

        if crowd_prompts is not None:
            cc_cfg_dict = dict(config.crowd_control_config or {})
            fake_cfg = SimpleNamespace(
                inject_mode=str(cc_cfg_dict.get("inject_mode", "append")),
                inject_source_mode=str(cc_cfg_dict.get("inject_source_mode", "as_image_source")),
                inject_label=str(cc_cfg_dict.get("inject_label", "Audience mutation request")),
                carryover_cycles=int(cc_cfg_dict.get("carryover_cycles", 0)),
            )
            fake_client = _FakeCrowdClient(crowd_prompts)
            fake_pkg = types.ModuleType("vfaq.crowd_control")
            fake_pkg.__path__ = []
            fake_models_mod = types.ModuleType("vfaq.crowd_control.models")
            fake_client_mod = types.ModuleType("vfaq.crowd_control.client")

            class _FakeCrowdControlConfig:
                @staticmethod
                def from_dict(_):
                    return fake_cfg

            def _build_client(_):
                return fake_client

            fake_models_mod.CrowdControlConfig = _FakeCrowdControlConfig
            fake_client_mod.CrowdClient = _build_client
            patchers.append(
                patch.dict(
                    sys.modules,
                    {
                        "vfaq.crowd_control": fake_pkg,
                        "vfaq.crowd_control.models": fake_models_mod,
                        "vfaq.crowd_control.client": fake_client_mod,
                    },
                )
            )

        with ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            result = run_sliding_story(
                story_path=self.story_path,
                qodeyard_dir=run_dir,
                config=config,
                max_cycles=max_cycles,
                base_image_path=(self.base_image if base_image else None),
            )
        return run_dir, result

    @staticmethod
    def _briq(run_dir: Path, cycle_idx: int) -> dict:
        path = run_dir / "briqs" / f"cycle_{cycle_idx:03d}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_default_behavior_unchanged_require_morph_true_and_false(self):
        config = self._make_story_config(require_morph=True, smart_enabled=False, text_to_video_first_cycle=True)
        main = _RecordingBackend("main")
        run_dir, _ = self._run_story(run_name="legacy_require_morph", config=config, main_backend=main, max_cycles=2)
        cycle2_image_idx = next(i for i, e in enumerate(main.events) if e["atom_id"] == "cycle_002_kf")
        cycle2_morph_idx = next(i for i, e in enumerate(main.events) if e["atom_id"] == "video_002" and e["kind"] == "generate_morph_video")
        self.assertLess(cycle2_image_idx, cycle2_morph_idx)
        self.assertEqual(len(main.morph_calls), 1)
        self.assertFalse(self._briq(run_dir, 2).get("smart_reinject_enabled"))

        config_no_morph = self._make_story_config(require_morph=False, smart_enabled=False, text_to_video_first_cycle=True)
        main_no_morph = _RecordingBackend("main_nomorph")
        self._run_story(run_name="legacy_no_morph", config=config_no_morph, main_backend=main_no_morph, max_cycles=2)
        self.assertEqual(len(main_no_morph.morph_calls), 0)
        self.assertTrue(any(atom == "video_002" for atom, _ in main_no_morph.video_calls))

    def test_smart_disabled_is_zero_impact(self):
        baseline_cfg = self._make_story_config(require_morph=True, smart_enabled=False, text_to_video_first_cycle=True)
        explicit_cfg = self._make_story_config(require_morph=True, smart_enabled=False, text_to_video_first_cycle=True)
        baseline = _RecordingBackend("baseline")
        explicit = _RecordingBackend("explicit")
        self._run_story(run_name="smart_off_baseline", config=baseline_cfg, main_backend=baseline, max_cycles=2)
        self._run_story(run_name="smart_off_explicit", config=explicit_cfg, main_backend=explicit, max_cycles=2)
        baseline_seq = [(e["kind"], e["atom_id"]) for e in baseline.events]
        explicit_seq = [(e["kind"], e["atom_id"]) for e in explicit.events]
        self.assertEqual(baseline_seq, explicit_seq)

    def test_smart_enabled_prefetch_does_not_block_video_start(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async", image_delay=0.7)
        started = time.time()
        self._run_story(
            run_name="smart_non_blocking",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=2,
        )
        elapsed = time.time() - started
        self.assertGreaterEqual(len(async_image.image_start_times), 1)
        self.assertGreaterEqual(len(main.video_call_times), 1)
        self.assertLess(elapsed, 0.7)

    def test_smart_enabled_next_cycle_uses_prefetched_keyframe_for_morph(self):
        config = self._make_story_config(
            require_morph=True,
            smart_enabled=True,
            smart_use_morph=True,
            text_to_video_first_cycle=False,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async")
        run_dir, _ = self._run_story(
            run_name="smart_use_prefetch",
            config=config,
            main_backend=main,
            async_backend=async_image,
            similarity_score=0.95,
            max_cycles=2,
        )
        cycle2_morph = [m for m in main.morph_calls if m[0] == "video_002"]
        self.assertEqual(len(cycle2_morph), 1)
        _, start_path, end_path = cycle2_morph[0]
        self.assertTrue(start_path.endswith("lastframe_001.png"))
        self.assertTrue(end_path.endswith("smart_reinject_target_002.png"))
        self.assertFalse(any(atom == "cycle_002_kf" for atom in main.image_calls))
        cycle2 = self._briq(run_dir, 2)
        self.assertTrue(cycle2.get("smart_reinject_used"))
        self.assertTrue(cycle2.get("smart_reinject_schedule_allowed"))
        self.assertTrue(cycle2.get("smart_reinject_consume_allowed"))
        self.assertIsNone(cycle2.get("smart_reinject_skip_reason"))

    def test_similarity_guard_reject_falls_back_to_image_to_video(self):
        config = self._make_story_config(
            require_morph=True,
            smart_enabled=True,
            smart_use_morph=True,
            text_to_video_first_cycle=False,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async")
        run_dir, _ = self._run_story(
            run_name="smart_similarity_reject",
            config=config,
            main_backend=main,
            async_backend=async_image,
            similarity_score=0.12,
            max_cycles=2,
        )
        self.assertFalse(any(atom == "video_002" for atom, _, _ in main.morph_calls))
        cycle2_video = [v for v in main.video_calls if v[0] == "video_002"][0]
        self.assertTrue(cycle2_video[1].endswith("lastframe_001.png"))
        briq = self._briq(run_dir, 2)
        self.assertIn("smart_reinject_rejected_keyframe", briq)
        self.assertFalse(briq.get("smart_reinject_used"))

    def test_enable_end_frame_morph_false_falls_back_without_crash(self):
        config = self._make_story_config(
            require_morph=True,
            smart_enabled=True,
            smart_use_morph=True,
            text_to_video_first_cycle=False,
            enable_end_frame_morph=False,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async")
        self._run_story(
            run_name="smart_morph_disabled",
            config=config,
            main_backend=main,
            async_backend=async_image,
            similarity_score=0.95,
            max_cycles=2,
        )
        self.assertFalse(any(atom == "video_002" for atom, _, _ in main.morph_calls))
        cycle2_video = [v for v in main.video_calls if v[0] == "video_002"][0]
        self.assertTrue(cycle2_video[1].endswith("lastframe_001.png"))

    def test_pending_future_not_ready_with_zero_timeout_skips_wait(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            smart_wait_timeout_sec=0,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async", image_delay=1.0)
        self._run_story(
            run_name="smart_pending_not_ready",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=2,
        )
        cycle2_video = [v for v in main.video_calls if v[0] == "video_002"][0]
        self.assertTrue(cycle2_video[1].endswith("lastframe_001.png"))

    def test_wait_timeout_zero_marks_prefetch_stale_and_allows_reschedule(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            smart_wait_timeout_sec=0,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async", image_delay=1.0)
        run_dir, _ = self._run_story(
            run_name="smart_stale_missed_target",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=3,
        )
        cycle2 = self._briq(run_dir, 2)
        self.assertTrue(cycle2.get("smart_reinject_pending_detected"))
        self.assertTrue(cycle2.get("smart_reinject_missed_target_cycle"))
        self.assertTrue(cycle2.get("smart_reinject_scheduled"))

    def test_append_mode_resolves_prompt_with_labeled_crowd_block(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=False,
            crowd_enabled=True,
            crowd_inject_mode="append",
            crowd_inject_source_mode="as_image_source",
        )
        main = _RecordingBackend("main")
        crowd_text = "crowd add shimmering paper koi"
        run_dir, _ = self._run_story(
            run_name="crowd_append_prompt_resolution",
            config=config,
            main_backend=main,
            max_cycles=1,
            crowd_prompts=[crowd_text],
        )
        cycle1_request = [r for r in main.video_requests if r["atom_id"] == "video_001"][0]
        self.assertIn("Para one.", cycle1_request["prompt"])
        self.assertIn("[AUDIENCE MUTATION REQUEST]", cycle1_request["prompt"])
        self.assertIn(crowd_text, cycle1_request["prompt"])
        cycle1 = self._briq(run_dir, 1)
        self.assertEqual(cycle1.get("requested_inject_mode"), "append")
        self.assertEqual(cycle1.get("crowd_prompt_origin"), "fresh")
        self.assertEqual(cycle1.get("authoritative_prompt_route"), "crowd_main_route")

    def test_replace_mode_resolves_prompt_to_crowd_only(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=False,
            crowd_enabled=True,
            crowd_inject_mode="replace",
            crowd_inject_source_mode="as_image_source",
        )
        main = _RecordingBackend("main")
        crowd_text = "crowd replace with molten chrome owls"
        run_dir, _ = self._run_story(
            run_name="crowd_replace_prompt_resolution",
            config=config,
            main_backend=main,
            max_cycles=1,
            crowd_prompts=[crowd_text],
        )
        cycle1_request = [r for r in main.video_requests if r["atom_id"] == "video_001"][0]
        self.assertEqual(cycle1_request["prompt"], crowd_text)
        self.assertNotIn("Para one.", cycle1_request["prompt"])
        cycle1 = self._briq(run_dir, 1)
        self.assertEqual(cycle1.get("requested_inject_mode"), "replace")
        self.assertEqual(cycle1.get("crowd_prompt_origin"), "fresh")

    def test_crowd_prompt_carryover_replays_for_configured_cycles(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=False,
            crowd_enabled=True,
            crowd_inject_mode="append",
            crowd_inject_source_mode="as_image_source",
            crowd_carryover_cycles=1,
        )
        main = _RecordingBackend("main")
        crowd_text = "carry this mutation into next beat"
        run_dir, _ = self._run_story(
            run_name="crowd_prompt_carryover",
            config=config,
            main_backend=main,
            max_cycles=2,
            crowd_prompts=[crowd_text],
        )
        cycle2_request = [r for r in main.video_requests if r["atom_id"] == "video_002"][0]
        self.assertIn(crowd_text, cycle2_request["prompt"])
        cycle1 = self._briq(run_dir, 1)
        cycle2 = self._briq(run_dir, 2)
        self.assertEqual(cycle1.get("crowd_prompt_origin"), "fresh")
        self.assertEqual(cycle2.get("crowd_prompt_origin"), "carryover")
        self.assertEqual(cycle2.get("crowd_carryover_remaining"), 0)

    def test_crowd_cycle_pauses_smart_schedule_for_current_cycle_only(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            crowd_enabled=True,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async")
        run_dir, _ = self._run_story(
            run_name="smart_crowd_suppression",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=2,
            crowd_prompts=["mutate this cycle"],
        )
        self.assertEqual(async_image.image_calls, [])
        cycle1 = self._briq(run_dir, 1)
        cycle2 = self._briq(run_dir, 2)
        self.assertTrue(cycle1.get("crowd_active"))
        self.assertFalse(cycle1.get("smart_reinject_schedule_allowed"))
        self.assertFalse(cycle1.get("smart_reinject_consume_allowed"))
        self.assertFalse(cycle1.get("smart_reinject_apply_allowed"))
        self.assertEqual(cycle1.get("smart_reinject_skip_reason"), "crowd_prompt_active")
        self.assertFalse(cycle1.get("smart_reinject_scheduled"))
        self.assertTrue(config.smart_reinject_enabled)
        self.assertFalse(cycle2.get("crowd_active"))
        self.assertTrue(cycle2.get("smart_reinject_schedule_allowed"))
        self.assertTrue(cycle2.get("smart_reinject_consume_allowed"))
        self.assertTrue(cycle2.get("smart_reinject_apply_allowed"))

    def test_crowd_cycle_discards_pending_prefetch_and_skips_consumption(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            crowd_enabled=True,
            smart_wait_timeout_sec=0,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async", image_delay=1.0)
        run_dir, _ = self._run_story(
            run_name="smart_crowd_discard_pending",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=2,
            crowd_prompts=[None, "crowd cycle two prompt"],
        )
        cycle2 = self._briq(run_dir, 2)
        self.assertTrue(cycle2.get("crowd_active"))
        self.assertFalse(cycle2.get("smart_reinject_consume_allowed"))
        self.assertTrue(cycle2.get("smart_reinject_pending_detected"))
        self.assertTrue(cycle2.get("smart_reinject_discarded_due_to_crowd"))
        self.assertFalse(cycle2.get("smart_reinject_used"))
        self.assertEqual(cycle2.get("smart_reinject_skip_reason"), "crowd_prompt_active")
        cycle2_video = [v for v in main.video_calls if v[0] == "video_002"][0]
        self.assertTrue(cycle2_video[1].endswith("lastframe_001.png"))
        self.assertNotIn("smart_reinject_target_002.png", cycle2_video[1])
        self.assertTrue(config.smart_reinject_enabled)

    def test_smart_reinject_resumes_after_crowd_cycle(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            crowd_enabled=True,
            smart_wait_timeout_sec=0,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async")
        run_dir, _ = self._run_story(
            run_name="smart_resume_after_crowd",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=4,
            crowd_prompts=[None, "crowd cycle two prompt"],
        )
        cycle2 = self._briq(run_dir, 2)
        cycle3 = self._briq(run_dir, 3)
        self.assertTrue(cycle2.get("crowd_active"))
        self.assertFalse(cycle2.get("smart_reinject_schedule_allowed"))
        self.assertFalse(cycle2.get("smart_reinject_consume_allowed"))
        self.assertEqual(cycle2.get("smart_reinject_skip_reason"), "crowd_prompt_active")
        self.assertFalse(cycle3.get("crowd_active"))
        self.assertTrue(cycle3.get("smart_reinject_schedule_allowed"))
        self.assertTrue(cycle3.get("smart_reinject_consume_allowed"))
        self.assertTrue(cycle3.get("smart_reinject_scheduled"))
        self.assertIn("smart_reinject_003_for_004", async_image.image_calls)
        self.assertNotIn("smart_reinject_002_for_003", async_image.image_calls)

    def test_crowd_prompt_still_reaches_video_route_when_smart_paused(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            crowd_enabled=True,
            smart_wait_timeout_sec=0,
        )
        main = _RecordingBackend("main")
        async_image = _RecordingBackend("async", image_delay=1.0)
        crowd_text = "crowd tiger made of glass and rain"
        run_dir, _ = self._run_story(
            run_name="smart_crowd_prompt_reaches_video",
            config=config,
            main_backend=main,
            async_backend=async_image,
            max_cycles=2,
            crowd_prompts=[None, crowd_text],
        )
        cycle2_video_request = [r for r in main.video_requests if r["atom_id"] == "video_002"][0]
        self.assertIn(crowd_text, cycle2_video_request["prompt"])
        self.assertIn(crowd_text, cycle2_video_request["video_prompt"])
        self.assertTrue(cycle2_video_request["source"].endswith("lastframe_001.png"))
        cycle2 = self._briq(run_dir, 2)
        self.assertFalse(cycle2.get("smart_reinject_used"))
        self.assertEqual(cycle2.get("smart_reinject_skip_reason"), "crowd_prompt_active")

    def test_as_reference_unsupported_falls_back_to_as_image_source_with_truthful_metadata(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=False,
            text_to_video_first_cycle=False,
            crowd_enabled=True,
            crowd_inject_mode="append",
            crowd_inject_source_mode="as_reference",
        )
        main = _ReferenceStripThenI2VBackend("main")
        run_dir, _ = self._run_story(
            run_name="crowd_as_reference_fallback_to_i2v",
            config=config,
            main_backend=main,
            max_cycles=2,
            crowd_prompts=[None, "crowd crystal lotus in stormlight"],
        )
        cycle2 = self._briq(run_dir, 2)
        self.assertEqual(cycle2.get("requested_inject_source_mode"), "as_reference")
        self.assertEqual(cycle2.get("actual_inject_source_mode"), "as_image_source")
        self.assertTrue(cycle2.get("reference_images_requested"))
        self.assertFalse(cycle2.get("reference_images_accepted"))
        self.assertTrue(cycle2.get("reference_images_stripped_by_retry"))
        self.assertTrue(cycle2.get("as_reference_fallback_to_image_source"))
        cycle2_calls = [v for v in main.video_calls if v[0] == "video_002"]
        self.assertEqual(len(cycle2_calls), 2)
        self.assertIsNone(cycle2_calls[0][1])
        self.assertTrue(cycle2_calls[1][1].endswith("lastframe_001.png"))

    def test_interrupt_cleanup_helper_shuts_down_active_prefetch_workers(self):
        config = self._make_story_config(
            require_morph=False,
            smart_enabled=True,
            text_to_video_first_cycle=False,
            smart_wait_timeout_sec=0,
        )
        main = _InterruptingBackend("main", interrupt_atom_id="video_002")
        async_image = _RecordingBackend("async", image_delay=1.5)
        with self.assertRaises(KeyboardInterrupt):
            self._run_story(
                run_name="smart_interrupt_cleanup",
                config=config,
                main_backend=main,
                async_backend=async_image,
                max_cycles=3,
            )
        shutdown_active_smart_reinject_workers(reason="interrupt")
        shutdown_active_smart_reinject_workers(reason="interrupt")

    def test_briq_metadata_contains_smart_fields_for_scheduled_used_and_rejected(self):
        use_cfg = self._make_story_config(
            require_morph=True,
            smart_enabled=True,
            smart_use_morph=True,
            text_to_video_first_cycle=False,
        )
        main_use = _RecordingBackend("main_use")
        async_use = _RecordingBackend("async_use")
        run_dir_use, _ = self._run_story(
            run_name="smart_meta_used",
            config=use_cfg,
            main_backend=main_use,
            async_backend=async_use,
            similarity_score=0.95,
            max_cycles=2,
        )
        cycle1 = self._briq(run_dir_use, 1)
        cycle2 = self._briq(run_dir_use, 2)
        self.assertTrue(cycle1.get("smart_reinject_scheduled"))
        self.assertEqual(cycle1.get("smart_reinject_scheduled_for_cycle"), 2)
        self.assertTrue(cycle2.get("smart_reinject_enabled"))
        self.assertTrue(cycle2.get("smart_reinject_used"))
        self.assertTrue(cycle2.get("smart_reinject_keyframe"))

        reject_cfg = self._make_story_config(
            require_morph=True,
            smart_enabled=True,
            smart_use_morph=True,
            text_to_video_first_cycle=False,
        )
        main_reject = _RecordingBackend("main_reject")
        async_reject = _RecordingBackend("async_reject")
        run_dir_reject, _ = self._run_story(
            run_name="smart_meta_rejected",
            config=reject_cfg,
            main_backend=main_reject,
            async_backend=async_reject,
            similarity_score=0.05,
            max_cycles=2,
        )
        cycle2_reject = self._briq(run_dir_reject, 2)
        self.assertIn("smart_reinject_rejected_keyframe", cycle2_reject)
        self.assertFalse(cycle2_reject.get("smart_reinject_used"))


if __name__ == "__main__":
    unittest.main()
