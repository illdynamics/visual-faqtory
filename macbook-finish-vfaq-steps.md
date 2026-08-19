# MacBook Finish Steps — Local MLX backend (verified)

The `local` backend is wired into Visual FaQtory and has been verified on this
MacBook (Apple Silicon) with **mflux** for the image models. Wan 2.2 video uses
the SceneWorks native worker (`mlx-gen-wan`), which is a separate tool.

> ⚠️ The old instructions in this file assumed a `lightning-mlx generate-image`
> CLI. That tool is an LLM server, not an image generator, so the backend was
> reimplemented. Use `mflux` instead.

## 0. What works out of the box

- `backend.type: local` creates a `LocalBackend`.
- Model selection: `flux-1-dev`, `flux-1-kontext`, `wan-2.2`, `z-image-turbo`.
- Runner selection: `mflux` (recommended images), `flux-swift`, `wan`
  (SceneWorks), or a fully-custom `command`.
- `python vfaq_cli.py backends` checks runner executables and reports per-model
  availability.

## 1. Install the image runner (mflux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt mflux
```

Confirm the CLIs exist:

```bash
which mflux-generate mflux-generate-z-image-turbo mflux-generate-kontext
```

## 2. Configure the models

```bash
cp worqspace/config-local.example.yaml worqspace/config.yaml
```

`local.model_paths` values are passed to the runner's `--model` flag. For
`mflux` they may be a local path, an mflux-compatible HuggingFace repo id
(`org/name`), or a built-in mflux model name. Verified working sources:

```yaml
local:
  runner: mflux
  quantize: 4
  models:
    image: z-image-turbo
    video: wan-2.2
    edit: flux-1-kontext
  model_paths:
    z-image-turbo: filipstrand/Z-Image-Turbo-mflux-4bit
    flux-1-dev: dhairyashil/FLUX.1-dev-mflux-4bit
    flux-1-kontext: akx/FLUX.1-Kontext-dev-mflux-4bit
    wan-2.2: ~/Qoding/ai/wan2.2-ti2v-5b-mlx
```

> Your `~/Qoding/ai` checkpoints are three different formats:
> `flux1.dev.4bit.mlx` / `flux1.kontext.4bit.mlx` (flux.swift),
> `Z-Image-Turbo-MLX-4bit` (diffusers-MLX), `wan2.2-ti2v-5b-mlx` (SceneWorks).
> Those are **not** mflux-readable. To run those exact files, install the
> matching tool and set `local.model_runners`:
>
> ```yaml
> local:
>   model_runners:
>     flux-1-dev: flux-swift      # flux.swift.cli
>     flux-1-kontext: flux-swift
>     wan-2.2: wan                # mlx-gen-wan
> ```

## 3. Verify availability

```bash
python vfaq_cli.py backends
```

Expected (with mflux installed and Wan not yet installed):

```
  [✓] local - Local backend partially available — ready: Z-Image-Turbo, FLUX.1 Kontext; missing: Wan 2.2: Runner executable not found on PATH: mlx-gen-wan
```

## 4. Smoke-test the image models

```bash
python - <<'PY'
from pathlib import Path
import yaml
from vfaq.backends import create_backend, GenerationRequest, InputMode

cfg = yaml.safe_load(Path('worqspace/config-local.example.yaml').read_text())
# point image models at mflux-compatible sources for a quick test
cfg['local']['model_paths']['z-image-turbo'] = 'filipstrand/Z-Image-Turbo-mflux-4bit'
cfg['local']['model_paths']['flux-1-dev'] = 'dhairyashil/FLUX.1-dev-mflux-4bit'
cfg['local']['model_paths']['flux-1-kontext'] = 'akx/FLUX.1-Kontext-dev-mflux-4bit'
backend = create_backend(cfg)
out = Path('run/_local_smoke'); out.mkdir(parents=True, exist_ok=True)
img = backend.generate_image(GenerationRequest(
    prompt='a neon jellyfish drifting through fog',
    mode=InputMode.TEXT, width=512, height=512, steps=4, output_dir=out, atom_id='smoke'))
print(img.success, img.error, img.image_path)
PY
```

## 5. Wan 2.2 video

Wan 2.2 TI2V requires the SceneWorks native worker (`mlx-gen-wan`). It is not
pip-installable. Install it and place `mlx-gen-wan` on your PATH, then set:

```yaml
local:
  wan:
    executable: mlx-gen-wan
```

Until then the backend reports Wan as unavailable, and morph requests fall back
to an ffmpeg crossfade so image-only runs can still complete.
