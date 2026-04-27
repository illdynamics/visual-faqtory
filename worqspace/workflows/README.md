# ComfyUI workflow expectations

Visual FaQtory does **not** bundle large ComfyUI workflow JSONs. The filenames referenced by the example configs are operator-supplied API-format workflow exports that must match the node packs, checkpoints, and motion models installed on your actual ComfyUI machine.

Recommended files to place here:
- `qwen_image_t2i.json` — Qwen text-to-image workflow
- `qwen_image_i2i.json` — Qwen image-to-image workflow with at least one `LoadImage` node
- `svd_img2vid.json` — ComfyUI SVD image-to-video workflow
- `morph_i2v.json` — explicit two-image morph / loop-closure workflow with at least two `LoadImage` nodes
- `animatediff_i2v.json` — AnimateDiff image-to-video workflow (optional; the backend can build a default one)
- `animatediff_morph_i2v.json` — bundled starter two-image AnimateDiff morph workflow (required for require_morph / loop closure if morph_backend=animatediff)

Runtime injection rules:
- Prompts are injected into whichever conditioning nodes feed `KSampler` positive / negative.
- Width / height are injected into nodes exposing `width` + `height`, or `megapixels` on `ImageScaleToTotalPixels`.
- Init images are uploaded into `LoadImage` nodes.
- `workflow_morph` is never assumed automatically; configure it explicitly.

AnimateDiff workflow contract:
- `workflow_video` must be an API-format ComfyUI JSON, not the full UI export.
- For img2vid, the workflow must contain at least one `LoadImage` node and one real video output node such as `VHS_VideoCombine`.
- To let Visual FaQtory inject AnimateDiff config values automatically, the workflow should include an AnimateDiff Evolved loader node such as `ADE_AnimateDiffLoaderGen1`.
- Context controls are injected only when the workflow exposes AnimateDiff context nodes such as `ADE_AnimateDiffUniformContextOptions` or `ADE_StandardStaticContextOptions`.
- Motion LoRA injection is only attempted when the workflow (or installed node pack) supports `ADE_AnimateDiffLoRALoader`.
- Prompt travel / prompt schedule is best-effort only and requires schedule nodes already present in the workflow.

- If `motion_loras` are configured, the workflow must expose a compatible AnimateDiff loader with a `motion_lora` input; otherwise Visual FaQtory now fails fast with a config error.

Known-good workflow contract summary:
- **Qwen text2img**: API-format JSON for still-image generation; required as `workflow_image` for `qwen_image_comfyui`.
- **Qwen img2img**: same, plus at least one `LoadImage` node for reinject/resume.
- **SVD img2vid**: API-format JSON with one `LoadImage` start-frame input and a real video output node/path.
- **AnimateDiff img2vid**: API-format JSON with `LoadImage`, an AnimateDiff loader node, and a real video output node such as `VHS_VideoCombine`, unless you rely on the backend's built-in default graph.
- **AnimateDiff morph**: explicit two-image AnimateDiff workflow with two image inputs. This repo now includes a starter graph that seeds the first frame from image A and the last frame from image B using `ReplaceVideoLatentFrames`, then lets AnimateDiff fill the motion in between.
- **ComfyUI morph**: explicit two-image morph / loop-closure workflow with two `LoadImage` nodes and a real video output path.

Troubleshooting bad or missing graphs:
- Export **API format JSON**, not the normal UI workflow save.
- If ComfyUI cannot import the graph cleanly, Visual FaQtory cannot patch it cleanly either.
- Missing `LoadImage`, AnimateDiff loader, or video-output nodes are treated as config errors.
- Keep checkpoint, motion model, and optional motion LoRA names aligned with what ComfyUI actually exposes.
