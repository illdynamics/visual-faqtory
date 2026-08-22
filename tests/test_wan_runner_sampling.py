from pathlib import Path

import pytest

from vfaq.backends import GenerationRequest
from vfaq.local_backend import WanRunner


def _build(config, *, source_image=None):
    request = GenerationRequest(
        prompt="a lighthouse at dusk",
        output_dir=Path("/tmp"),
        video_frames=65,
        video_fps=12,
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


def _sampling_flags(cmd):
    flags = {}
    for idx, arg in enumerate(cmd):
        if arg == "--denoising-step-list":
            values = []
            j = idx + 1
            while j < len(cmd) and cmd[j].lstrip("-").isdigit():
                values.append(cmd[j])
                j += 1
            flags[arg] = values
        elif arg in ("--steps", "--flow-shift"):
            flags[arg] = cmd[idx + 1]
    return flags


def test_step_mode_emits_steps_and_flow_shift():
    cmd = _build({"steps": 4, "flow_shift": 3.0})
    flags = _sampling_flags(cmd)
    assert flags["--steps"] == "4"
    assert flags["--flow-shift"] == "3.0"
    assert "--denoising-step-list" not in flags


def test_grid_mode_emits_only_denoising_step_list():
    cmd = _build({"denoising_step_list": [1000, 750, 500, 250], "steps": "off", "flow_shift": "off"})
    flags = _sampling_flags(cmd)
    assert flags["--denoising-step-list"] == ["1000", "750", "500", "250"]
    assert "--steps" not in flags
    assert "--flow-shift" not in flags


def test_all_off_emits_no_sampling_flags():
    cmd = _build({"denoising_step_list": "off", "steps": 0, "flow_shift": "off"})
    flags = _sampling_flags(cmd)
    assert flags == {}


def test_grid_mode_supports_space_separated_string():
    cmd = _build({"denoising-step-list": "1000 750 500 250", "steps": "none", "flow-shift": "none"})
    flags = _sampling_flags(cmd)
    assert flags["--denoising-step-list"] == ["1000", "750", "500", "250"]
    assert "--steps" not in flags
    assert "--flow-shift" not in flags


def test_conflicting_grid_and_steps_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build({"denoising_step_list": [1000], "steps": 4})


def test_conflicting_grid_and_flow_shift_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build({"denoising_step_list": [1000], "flow_shift": 3.0})
