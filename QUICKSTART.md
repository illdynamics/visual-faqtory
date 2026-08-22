# Visual FaQtory — Quickstart Guide

This guide gets you from "I just cloned the repo" to "I have a finished video"
with as little jargon as possible. Pick one goal below and follow the numbered
steps.

---

## Goal A — Make a promo / music video from your own track

This is the most common use case: you have a song, you want visuals to go with it.

1. **Install** (from the repo folder):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (You also need `ffmpeg` on your system.)

2. **Put your inputs in `worqspace/`**:
   - Your track: `worqspace/base_audio/your-track.mp3`
   - A written story / mood text: `worqspace/story.txt`
   - (Optional) a cover / starting image: `worqspace/base_images/your-image.jpg`

3. **Pick a backend in `worqspace/config.yaml`**:
   - **Easiest (cloud):** set `backend.type: venice` and `export VENICE_API_KEY=...`
   - **On your MacBook (local, free, private):** set `backend.type: local`, then
     fill in `local.model_paths.*` (see `worqspace/configs/config-local.example.yaml`).

4. **Make the video length match your track** (optional but nice):
   In `worqspace/config.yaml`, under `audio:`, set:
   ```yaml
   audio:
     enabled: true
     sync_video_audio: true
     cycle_seconds: 5
   ```
   (Or use `bpm: 128` + `beats_per_cycle: 4` instead of `cycle_seconds`.)

5. **Run it**:
   ```bash
   python vfaq_cli.py run -n my-promo
   ```

6. **Get your video**:
   The finished file is saved in `worqspace/saved-runs/my-promo/my-promo.mp4`
   (and the higher-quality `final_60fps_1080p.mp4` / `*_audio.mp4` variants).

---

## Goal B — Live visuals with crowd control (QR prompts)

Use this when you are streaming / DJ-ing and want the audience to mutate the
visuals by scanning a QR code.

1. **Start the crowd-control web app** (on the visuals machine):
   ```bash
   export VF_CROWD_TOKEN=change-me-to-a-long-random-string
   python vfaq_cli.py crowd --host 0.0.0.0 --port 8000
   ```
   This prints the URL for the prompt page, QR code, and OBS overlay.

2. **In `worqspace/config.yaml`**, set:
   ```yaml
   crowd_control:
     enabled: true
     base_url: "http://127.0.0.1:8000/visuals"
     pop_token: "${VF_CROWD_TOKEN}"
   ```

3. **Run the generator** (a second terminal, same machine):
   ```bash
   export VF_CROWD_TOKEN=change-me-to-a-long-random-string
   python vfaq_cli.py run -n live-set
   ```

4. **Add the QR overlay in OBS**:
   Add a browser source pointing to `http://127.0.0.1:8000/visuals/qr.png`
   (or the public URL shown when you started the crowd server).

5. **Point OBS at the generated video** with the included watcher scripts
   (`vf-obs-watcher-same-machine.sh` / `vf-obs-watcher-srt-endpoints.sh`), and
   configure your `OBS_PASSWORD` environment variable.

---

## Goal C — Run locally on Apple Silicon (MacBook M-series)

1. Install the image runner (recommended):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt mflux
   ```
2. Copy the example config:
   ```bash
   cp worqspace/configs/config-local.example.yaml worqspace/config.yaml
   ```
3. Edit `worqspace/config.yaml` → `local.model_paths`. For the image models
   (`z-image-turbo`, `flux-1-dev`, `flux-1-kontext`) you can use an
   mflux-compatible HuggingFace repo id or a built-in name (e.g.
   `z-image-turbo`, `dev`, `dev-kontext`) and mflux will download it. For
   Wan 2.2 video you need SceneWorks `mlx-gen-wan` (a separate native tool).
4. Verify what is available:
   ```bash
   python vfaq_cli.py backends
   ```
5. Run:
   ```bash
   python vfaq_cli.py run -n local-test
   ```

> The local backend's image models (Z-Image-Turbo, FLUX.1-dev, FLUX.1 Kontext)
> have been verified on Apple Silicon via `mflux`. Wan 2.2 video requires the
> SceneWorks native worker (`mlx-gen-wan`); until it is installed the backend
> reports Wan as unavailable and morph falls back to an ffmpeg crossfade.

> Wan 2.2 sampling modes: `denoising_step_list`, `steps`, and `flow_shift` are
> mutually exclusive. Use either a Self-Forcing grid (`denoising_step_list`)
> or step mode (`steps` + optional `flow_shift`); set the unused keys to `off`
> (also `0` or `[]`). See `doc/DOCUMENTATION.md` → "Wan 2.2 sampling modes".

---

## Other modes / notes

- `python vfaq_cli.py status` — see whether a run is resumable.
- `python vfaq_cli.py run --dry-run` — validate config without generating.
- `python vfaq_cli.py run --resume` — continue a run from the last checkpoint.
- `python vfaq_cli.py backends` — check which backends are ready.

For the full reference, see `doc/DOCUMENTATION.md`.
