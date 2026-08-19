# ComfyUI workflow expectations

Visual FaQtory does **not** bundle large ComfyUI workflow JSONs. The filenames referenced by the example configs are operator-supplied API-format workflow exports that must match the node packs, checkpoints, and motion models installed on your actual ComfyUI machine.

Recommended files to place here:
- `sdxl_t2i.json` — SDXL text-to-image workflow (optional; backend can build a default)
- `sdxl_i2i.json` — SDXL image-to-image workflow with a `LoadImage` node (optional)
- `svd_img2vid.json` — ComfyUI SVD image-to-video workflow
- `morph_i2v.json` — explicit two-image morph / loop-closure workflow with at least two `LoadImage` nodes
- `animatediff_i2v.json` — AnimateDiff image-to-video workflow (optional; the backend can build a default one)

Runtime injection rules:
- Prompts are injected into whichever conditioning nodes feed `KSampler` positive / negative.
- Width / height are injected into nodes exposing `width` + `height`, or `megapixels` on `ImageScaleToTotalPixels`.
- Init images are uploaded into `LoadImage` nodes.
- `workflow_morph` is never assumed automatically; configure it explicitly.

AnimateDiff workflow contract:
- `workflow_video` must be an API-format ComfyUI JSON, not the full UI export.
- For img2vid, the workflow must contain at least one `LoadImage` node and one real video output node such as `VHS_VideoCombine`.
- Motion LoRA injection is only attempted when the workflow (or installed node pack) supports `ADE_AnimateDiffLoRALoader`.

Known-good workflow contract summary:
- **SDXL text2img**: API-format JSON for still-image generation (or omit to use the built-in default graph).
- **SVD img2vid**: API-format JSON with one `LoadImage` start-frame input and a real video output node/path.
- **AnimateDiff img2vid**: API-format JSON with `LoadImage`, an AnimateDiff loader node, and a real video output node such as `VHS_VideoCombine`, unless you rely on the backend's built-in default graph.
- **ComfyUI morph**: explicit two-image morph / loop-closure workflow with two `LoadImage` nodes and a real video output path.

Troubleshooting bad or missing graphs:
- Export **API format JSON**, not the normal UI workflow save.
- If ComfyUI cannot import the graph cleanly, Visual FaQtory cannot patch it cleanly either.
- Missing `LoadImage`, AnimateDiff loader, or video-output nodes are treated as config errors.
- Keep checkpoint, motion model, and optional motion LoRA names aligned with what ComfyUI actually exposes.
