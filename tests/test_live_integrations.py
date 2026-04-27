from __future__ import annotations

from pathlib import Path

from vfaq.backends import (
    AnimateDiffBackend,
    ComfyUIBackend,
    GenerationRequest,
    InputMode,
    QwenImageComfyUIBackend,
)
from vfaq.venice_backend import VeniceBackend

from tests.live_helpers import (
    assert_valid_image,
    assert_valid_video,
    build_venice_config,
    comfy_api_url,
    comfy_timeout,
    create_placeholder_png,
    optional_env,
    require_existing_file_env,
    require_env,
    skip_unless_flag,
)


def _smoke_prompt(prefix: str) -> str:
    return f"{prefix}. Minimal live smoke test, simple composition, cheap runtime."


class TestLiveComfyIntegrations:
    def test_live_qwen_text2img(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_COMFY_TESTS", "live ComfyUI tests")
        workflow_image = require_existing_file_env("VF_COMFY_QWEN_WORKFLOW_IMAGE", "Qwen text2img live test")

        backend = QwenImageComfyUIBackend({
            "api_url": comfy_api_url(),
            "timeout": comfy_timeout(),
            "workflow_image": str(workflow_image),
        })
        ok, message = backend.check_availability()
        assert ok, message

        output_dir = tmp_path / "qwen_text2img"
        result = backend.generate_image(
            GenerationRequest(
                prompt=_smoke_prompt("Qwen live text2img"),
                mode=InputMode.TEXT,
                width=512,
                height=512,
                steps=8,
                cfg_scale=4.0,
                output_dir=output_dir,
                atom_id="live_qwen_t2i",
            )
        )

        assert result.success, result.error
        media_type = assert_valid_image(result.image_path)
        assert media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}

    def test_live_qwen_img2img(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_COMFY_TESTS", "live ComfyUI tests")
        workflow_image = require_existing_file_env("VF_COMFY_QWEN_WORKFLOW_IMAGE", "Qwen text2img live test")
        workflow_img2img = require_existing_file_env("VF_COMFY_QWEN_WORKFLOW_IMG2IMG", "Qwen img2img live test")
        source_image = create_placeholder_png(tmp_path / "qwen_input.png")

        backend = QwenImageComfyUIBackend({
            "api_url": comfy_api_url(),
            "timeout": comfy_timeout(),
            "workflow_image": str(workflow_image),
            "workflow_img2img": str(workflow_img2img),
        })
        ok, message = backend.check_availability()
        assert ok, message

        output_dir = tmp_path / "qwen_img2img"
        result = backend.generate_image(
            GenerationRequest(
                prompt=_smoke_prompt("Qwen live img2img"),
                mode=InputMode.IMAGE,
                init_image_path=source_image,
                denoise_strength=0.35,
                width=512,
                height=512,
                steps=8,
                cfg_scale=4.0,
                output_dir=output_dir,
                atom_id="live_qwen_i2i",
            )
        )

        assert result.success, result.error
        media_type = assert_valid_image(result.image_path)
        assert media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}

    def test_live_comfy_svd_img2vid(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_COMFY_TESTS", "live ComfyUI tests")
        workflow_video = require_existing_file_env("VF_COMFY_SVD_WORKFLOW_VIDEO", "ComfyUI SVD img2vid live test")
        source_image = create_placeholder_png(tmp_path / "svd_input.png")

        backend = ComfyUIBackend({
            "api_url": comfy_api_url(),
            "timeout": comfy_timeout(),
            "workflow_video": str(workflow_video),
        })
        ok, message = backend.check_availability()
        assert ok, message

        output_dir = tmp_path / "svd_img2vid"
        result = backend.generate_video(
            GenerationRequest(
                prompt=_smoke_prompt("ComfyUI SVD live img2vid"),
                mode=InputMode.IMAGE,
                width=512,
                height=512,
                video_frames=8,
                video_fps=6,
                steps=8,
                cfg_scale=2.5,
                output_dir=output_dir,
                atom_id="live_svd_i2v",
            ),
            source_image,
        )

        assert result.success, result.error
        media_type = assert_valid_video(result.video_path)
        assert media_type in {"video/mp4", "video/webm"}

    def test_live_animatediff_img2vid(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_COMFY_TESTS", "live ComfyUI tests")
        workflow_video = require_existing_file_env("VF_COMFY_ANIMATEDIFF_WORKFLOW_VIDEO", "AnimateDiff img2vid live test")
        checkpoint = require_env("VF_COMFY_ANIMATEDIFF_CHECKPOINT", "AnimateDiff live test")
        motion_model = require_env("VF_COMFY_ANIMATEDIFF_MOTION_MODEL", "AnimateDiff live test")
        source_image = create_placeholder_png(tmp_path / "animatediff_input.png")

        animatediff_cfg = {
            "checkpoint": checkpoint,
            "motion_model": motion_model,
            "frame_rate": 6,
            "steps": 8,
            "cfg": 3.0,
            "denoise_strength": 0.65,
        }
        negative_prompt = optional_env("VF_COMFY_ANIMATEDIFF_NEGATIVE_PROMPT")
        if negative_prompt:
            animatediff_cfg["negative_prompt"] = negative_prompt

        backend = AnimateDiffBackend({
            "api_url": comfy_api_url(),
            "timeout": comfy_timeout(),
            "workflow_video": str(workflow_video),
            "animatediff": animatediff_cfg,
        })
        ok, message = backend.check_availability()
        assert ok, message

        output_dir = tmp_path / "animatediff_img2vid"
        result = backend.generate_video(
            GenerationRequest(
                prompt=_smoke_prompt("AnimateDiff live img2vid"),
                mode=InputMode.IMAGE,
                width=512,
                height=512,
                duration_seconds=2.0,
                video_fps=6,
                steps=8,
                cfg_scale=3.0,
                denoise_strength=0.65,
                output_dir=output_dir,
                atom_id="live_ad_i2v",
            ),
            source_image,
        )

        assert result.success, result.error
        media_type = assert_valid_video(result.video_path)
        assert media_type in {"video/mp4", "video/webm"}


class TestLiveVeniceIntegrations:
    def test_live_venice_auth_and_models(self):
        skip_unless_flag("VF_RUN_LIVE_VENICE_TESTS", "live Venice tests")
        backend = VeniceBackend(build_venice_config())
        ok, message = backend.check_availability()
        assert ok, message

    def test_live_venice_text2img(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_VENICE_TESTS", "live Venice tests")
        backend = VeniceBackend(build_venice_config())
        output_dir = tmp_path / "venice_text2img"
        result = backend.generate_image(
            GenerationRequest(
                prompt=_smoke_prompt("Venice live text2img"),
                mode=InputMode.TEXT,
                width=512,
                height=512,
                output_dir=output_dir,
                atom_id="live_venice_t2i",
            )
        )

        assert result.success, result.error
        media_type = assert_valid_image(result.image_path)
        assert media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}
        assert result.metadata.get("model")
        assert result.metadata.get("response", {}).get("id") is not None

    def test_live_venice_img2img(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_VENICE_TESTS", "live Venice tests")
        backend = VeniceBackend(build_venice_config())
        source_image = create_placeholder_png(tmp_path / "venice_img2img_input.png")
        output_dir = tmp_path / "venice_img2img"
        result = backend.generate_image(
            GenerationRequest(
                prompt=_smoke_prompt("Venice live img2img"),
                mode=InputMode.IMAGE,
                init_image_path=source_image,
                width=512,
                height=512,
                denoise_strength=0.35,
                output_dir=output_dir,
                atom_id="live_venice_i2i",
            )
        )

        assert result.success, result.error
        media_type = assert_valid_image(result.image_path)
        assert media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}
        assert result.metadata.get("model")
        assert result.metadata.get("operation") == "img2img"

    def test_live_venice_text2vid(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_VENICE_TESTS", "live Venice tests")
        backend = VeniceBackend(build_venice_config())
        output_dir = tmp_path / "venice_text2vid"
        result = backend.generate_video(
            GenerationRequest(
                prompt=_smoke_prompt("Venice live text2vid"),
                mode=InputMode.TEXT,
                duration_seconds=5.0,
                width=512,
                height=512,
                output_dir=output_dir,
                atom_id="live_venice_t2v",
            ),
            source_image=None,
        )

        assert result.success, result.error
        media_type = assert_valid_video(result.video_path)
        assert media_type in {"video/mp4", "video/webm"}
        assert result.metadata.get("queue_id")
        assert result.metadata.get("model")

    def test_live_venice_img2vid(self, tmp_path: Path):
        skip_unless_flag("VF_RUN_LIVE_VENICE_TESTS", "live Venice tests")
        backend = VeniceBackend(build_venice_config())
        source_image = create_placeholder_png(tmp_path / "venice_img2vid_input.png")
        output_dir = tmp_path / "venice_img2vid"
        result = backend.generate_video(
            GenerationRequest(
                prompt=_smoke_prompt("Venice live img2vid"),
                mode=InputMode.IMAGE,
                duration_seconds=5.0,
                width=512,
                height=512,
                output_dir=output_dir,
                atom_id="live_venice_i2v",
            ),
            source_image=source_image,
        )

        assert result.success, result.error
        media_type = assert_valid_video(result.video_path)
        assert media_type in {"video/mp4", "video/webm"}
        assert result.metadata.get("queue_id")
        assert result.metadata.get("model")
