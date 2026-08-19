# MacBook Finish Steps — Local MLX backend

The `local` backend is implemented and wired into Visual FaQtory, but it has
**not been executed against real model files** because the models live on your
MacBook (M1 Max) and use `lightning-mlx` (MLX). Everything below is what you run
on the MacBook to finish wiring + verify.

## 0. What already works from the repo

- `backend.type: local` creates a `LocalBackend`.
- Model selection: `flux-1-dev`, `flux-1-kontext`, `wan-2.2`, `z-image-turbo`.
- Runner selection: `lightning-mlx` (default), `mflux`, or fully-custom `command`.
- `python vfaq_cli.py backends` checks the runner executable and model paths.
- The sliding-story engine already treats `local` like the generic ComfyUI path
  (text2img → img2img keyframe → img2vid / morph).

## 1. Install / locate the runner

```bash
which lightning-mlx || echo "lightning-mlx not on PATH"
```

If it is not installed, install it with your package manager of choice, e.g.:

```bash
# Replace with the real install command for your setup
pipx install lightning-mlx      # or: pip install lightning-mlx
```

Confirm the CLI surface:

```bash
lightning-mlx --help
```

The defaults in `vfaq/local_backend.py` assume subcommands like:

```bash
lightning-mlx generate-image --model ... --model-path ... --prompt ... --output ... ...
lightning-mlx generate-video --model ... --model-path ... --prompt ... --image ... --output ... ...
```

**If your installed CLI uses different subcommand/flag names**, do not edit code —
override the templates in `worqspace/config.yaml`:

```yaml
local:
  runner: lightning-mlx
  lightning_mlx:
    executable: lightning-mlx
    image_template: "lightning-mlx <real-image-command> --prompt {prompt} --output {output} ..."
    video_template: "lightning-mlx <real-video-command> --prompt {prompt} --image {input_image} --output {output} ..."
    morph_template: "lightning-mlx <real-morph-command> --start {start_image} --end {end_image} --output {output} ..."
```

Supported placeholders:
`{model} {model_path} {prompt} {negative_prompt} {output} {width} {height} {seed} {steps} {fps} {duration} {input_image} {start_image} {end_image}`

## 2. Point to the models on disk

Copy the local example config:

```bash
cp worqspace/config-local.example.yaml worqspace/config.yaml
```

Edit `worqspace/config.yaml` → `local.model_paths`:

```yaml
local:
  runner: lightning-mlx
  models:
    image: z-image-turbo      # or flux-1-dev
    video: wan-2.2
    edit: flux-1-kontext
  model_paths:
    flux-1-dev: /absolute/path/to/flux-1-dev
    flux-1-kontext: /absolute/path/to/flux-1-kontext
    wan-2.2: /absolute/path/to/wan-2.2
    z-image-turbo: /absolute/path/to/z-image-turbo
```

`model_paths` values may be a directory (HF checkout / MLX conversion) or a
single checkpoint file — use whatever `lightning-mlx` expects for that model.

## 3. Verify availability

```bash
python vfaq_cli.py backends
```

You want the `local` line to show:

```
  [✓] local        - Local backend ready via lightning-mlx — image=z-image-turbo, video=wan-2.2, edit=flux-1-kontext
```

If it shows missing paths, fix `local.model_paths`.

## 4. Smoke-test one image and one video

First, a direct unit-level check:

```bash
python - <<'PY'
from pathlib import Path
import yaml
from vfaq.backends import create_backend, GenerationRequest, InputMode

cfg = yaml.safe_load(Path('worqspace/config.yaml').read_text())
backend = create_backend(cfg)
print('availability:', backend.check_availability())

out = Path('run/_local_smoke')
out.mkdir(parents=True, exist_ok=True)

img = backend.generate_image(GenerationRequest(
    prompt='a neon jellyfish drifting through fog',
    mode=InputMode.TEXT,
    width=512,
    height=512,
    steps=4,
    output_dir=out,
    atom_id='smoke_img',
))
print('image:', img.success, img.error, img.image_path)

vid = backend.generate_video(GenerationRequest(
    prompt='slow camera drift, shimmering neon',
    mode=InputMode.IMAGE,
    width=512,
    height=512,
    duration_seconds=2.0,
    video_fps=6,
    output_dir=out,
    atom_id='smoke_vid',
), source_image=img.image_path)
print('video:', vid.success, vid.error, vid.video_path)
PY
```

## 5. Run a full short story

```bash
# keep it short for the first test
printf 'Paragraph one.\n\nParagraph two.' > worqspace/story.txt
python vfaq_cli.py run -n local-smoke
```

Then check `run/` for `video_*.mp4` and `worqspace/saved-runs/local-smoke/`.

## 6. If lightning-mlx is not the right tool

The backend also supports `mflux` (FLUX images) and a fully custom `command`
runner. For `mflux`:

```yaml
local:
  runner: mflux
  mflux:
    executable: mflux-generate
```

For a totally custom command (any local inference script):

```yaml
local:
  runner: command
  command:
    executable: my-local-generator
    image_template: "my-local-generator image --prompt {prompt} --output {output} ..."
    video_template: "my-local-generator video --prompt {prompt} --image {input_image} --output {output} ..."
    morph_template: "my-local-generator morph --start {start_image} --end {end_image} --output {output} ..."
```

## 7. If you get stuck

- Run with `PYTHONPATH=. python vfaq_cli.py run -n debug --dry-run` to isolate
  config/availability problems from generation problems.
- The exact command built by the backend is logged with a `[Local] Running:` line.
- Report that line + the exit output; the fix will almost always be a
  `*_template` override in `worqspace/config.yaml`.
