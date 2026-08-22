"""Tests for the wan runner's canvas_policy and frames configuration knobs."""
from pathlib import Path

import pytest

from vfaq.backends import GenerationRequest
from vfaq.local_backend import WanRunner


def _flag(cmd, name):
    try:
        idx = cmd.index(name)
    except ValueError:
        return None
    return cmd[idx + 1]


def _build(config, *, source_image=None, video_frames=None, video_fps=12):
    request = GenerationRequest(
        prompt="a lighthouse at dusk",
        output_dir=Path("/tmp"),
        video_frames=video_frames,
        video_fps=video_fps,
    )
    runner = WanRunner(config, local_cfg={"steps": 8, "width": 640, "height": 352, "fps": 12})
    return runner.build_video_command(
        "wan-2.2",
        "/tmp/wan-checkpoint",
        request,
        Path("/tmp/video.mp4"),
        source_image,
        {},
    )


def test_frames_config_directly_sets_frames_flag():
    cmd = _build({"frames": 121})
    assert _flag(cmd, "--frames") == "121"


def test_max_frames_is_legacy_alias_for_frames():
    cmd = _build({"max_frames": 121})
    assert _flag(cmd, "--frames") == "121"


def test_request_video_frames_wins_over_config():
    cmd = _build({"frames": 121}, video_frames=65)
    assert _flag(cmd, "--frames") == "65"


def test_canvas_policy_source_aspect_for_i2v():
    cmd = _build({"canvas_policy": "source-aspect"}, source_image=Path("/tmp/source.png"))
    assert _flag(cmd, "--canvas-policy") == "source-aspect"


def test_canvas_policy_defaults_to_exact_resize_for_i2v():
    cmd = _build({}, source_image=Path("/tmp/source.png"))
    assert _flag(cmd, "--canvas-policy") == "exact-resize"


def test_canvas_policy_invalid_value_raises():
    with pytest.raises(ValueError, match="canvas_policy"):
        _build({"canvas_policy": "banana"}, source_image=Path("/tmp/source.png"))


def test_canvas_policy_omitted_for_t2v():
    cmd = _build({"canvas_policy": "source-aspect"})
    assert "--canvas-policy" not in cmd
