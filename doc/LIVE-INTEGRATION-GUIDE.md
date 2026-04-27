# Visual FaQtory v0.9.0-beta — live integration guide

This repo now ships an **opt-in** live integration harness for external systems. The default offline suite stays fast and does **not** require ComfyUI, Venice, Qwen workflows, or paid API calls.

## What this harness covers

The gated live pytest module is `tests/test_live_integrations.py`.

It can exercise:
- Qwen image via ComfyUI (`qwen_image_comfyui`) — text2img and img2img
- ComfyUI SVD img2vid
- AnimateDiff via ComfyUI img2vid
- Venice text2img
- Venice img2img
- Venice text2vid
- Venice img2vid

## Safety rails

- Live tests are skipped unless you explicitly enable them.
- Missing env vars or missing workflow files produce **clear skip reasons**.
- The harness uses tiny generated PNG inputs and short prompts to keep runtime and spend down.
- Venice video jobs still cost credits. Treat the live Venice video tests as billable.

## Required gates

### ComfyUI / Qwen / AnimateDiff live tests

Set:

```bash
export VF_RUN_LIVE_COMFY_TESTS=1
```

Recommended / required env vars:

```bash
export VF_COMFYUI_API_URL=http://127.0.0.1:8188
export VF_COMFY_TIMEOUT=300

# Required for live Qwen text2img
export VF_COMFY_QWEN_WORKFLOW_IMAGE=/absolute/path/to/qwen_text2img_api.json

# Required for live Qwen img2img
export VF_COMFY_QWEN_WORKFLOW_IMG2IMG=/absolute/path/to/qwen_img2img_api.json

# Required for live SVD img2vid
export VF_COMFY_SVD_WORKFLOW_VIDEO=/absolute/path/to/svd_img2vid_api.json

# Required for live AnimateDiff img2vid
export VF_COMFY_ANIMATEDIFF_WORKFLOW_VIDEO=/absolute/path/to/animatediff_img2vid_api.json
export VF_COMFY_ANIMATEDIFF_CHECKPOINT=your_base_checkpoint.safetensors
export VF_COMFY_ANIMATEDIFF_MOTION_MODEL=your_motion_model.ckpt

# Optional
export VF_COMFY_ANIMATEDIFF_NEGATIVE_PROMPT="low quality, blurry"
```

### Venice live tests

Set:

```bash
export VF_RUN_LIVE_VENICE_TESTS=1
export VENICE_API_KEY=...
```

Optional / recommended env vars:

```bash
export VF_VENICE_BASE_URL=https://api.venice.ai/api/v1
export VF_VENICE_POLL_INTERVAL=5
export VF_VENICE_POLL_TIMEOUT=900

export VF_VENICE_MODEL_TEXT2IMG=z-image-turbo
export VF_VENICE_MODEL_IMG2IMG=qwen-edit
export VF_VENICE_MODEL_TEXT2VID=wan-2.5-preview-text-to-video
export VF_VENICE_MODEL_IMG2VID=wan-2.5-preview-image-to-video

export VF_VENICE_IMAGE_WIDTH=512
export VF_VENICE_IMAGE_HEIGHT=512
export VF_VENICE_IMAGE_CFG_SCALE=7.0
export VF_VENICE_IMAGE_STEPS=8

export VF_VENICE_VIDEO_DURATION_SECONDS=5
export VF_VENICE_VIDEO_RESOLUTION=720p
export VF_VENICE_VIDEO_ASPECT_RATIO=16:9
export VF_VENICE_VIDEO_AUDIO=0
```

## Exact commands

Offline default suite:

```bash
pytest -q
```

Run only the live harness file:

```bash
pytest -q tests/test_live_integrations.py -rs
```

Run only the ComfyUI live checks:

```bash
pytest -q tests/test_live_integrations.py -k "comfy or qwen or animatediff" -rs
```

Run only the Venice live checks:

```bash
pytest -q tests/test_live_integrations.py -k "venice" -rs
```

## Workflow requirements for ComfyUI

ComfyUI workflow JSONs are **operator-supplied**. This repo does not bundle known-good production JSONs.

The live harness expects API-exported JSON workflow files that already match your installed node stack and models.

Minimum expectations:
- Qwen text2img workflow: must be a valid API JSON graph for still-image generation.
- Qwen img2img workflow: must include at least one `LoadImage` node so the harness can inject the source image.
- SVD img2vid workflow: must accept one uploaded source image and produce a downloadable video output.
- AnimateDiff img2vid workflow: must include a real AnimateDiff loader node and produce a downloadable video output.

If your graph names, checkpoints, motion models, or custom nodes differ from the env values above, the live tests will fail honestly.

## What success looks like

### ComfyUI / Qwen / AnimateDiff
- the configured workflow path exists
- ComfyUI is reachable
- source-image upload works where relevant
- queued execution completes
- output downloads to the pytest temp directory
- downloaded file is non-empty and detected as image or video media

### Venice
- auth works
- model validation / listing works if enabled
- image requests return valid image media
- video queue + poll completes or times out clearly
- downloaded outputs are valid media
- metadata includes at least the selected model and, for video, the Venice `queue_id`

## Failure interpretation

- **Skipped** — env gate disabled, required env var missing, or required workflow path missing.
- **Failed before queueing** — bad config, wrong workflow path, missing ComfyUI custom nodes, missing models, or bad auth.
- **Failed during polling** — remote service accepted the request but did not finish successfully before timeout.
- **Failed media assertion** — the remote service returned something, but it was not valid downloadable image/video output.

## Cost / runtime notes

- Venice image and video tests may consume credits. Use them deliberately.
- Venice video tests are the most expensive part of the harness.
- ComfyUI tests still consume local GPU time.
- Keep prompts simple and durations tiny unless you deliberately want a deeper live burn-in.

## Honest limitation

This guide documents how to run the live checks, but this repo version does **not** claim those live checks were executed in the packaging pass unless a separate validation note explicitly says so.
