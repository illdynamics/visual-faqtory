# EXTERNAL-LIVE-VISUALS-SETUP.md

## Visual FaQtory — External SRT Live Visuals + Crowd Queue Overlay

### What this is

A guide for deploying Visual FaQtory in a split-box architecture where the GPU
generation machine and the OBS streaming machine are separate, connected over
ZeroTier (or any routed network).

### Architecture

```
┌─────────────────────────────────────┐     ZeroTier / LAN     ┌─────────────────────────────────┐
│         GPU SERVER                  │◄───────────────────────►│         OBS MACHINE             │
│                                     │                         │                                 │
│  Visual FaQtory + ComfyUI           │                         │  OBS Studio                     │
│  ├── vfaq_cli.py (generator)        │                         │  ├── Media Source A (SRT caller) │
│  ├── vfaq_cli.py crowd (FastAPI)    │     SRT port 9998 ────► │  │   srt://10.x:9998?mode=caller│
│  ├── vf-obs-watcher-srt-endpoints.sh│     SRT port 9999 ────► │  ├── Media Source B (SRT caller) │
│  │   ├── Slot A ffmpeg (listener)   │                         │  │   srt://10.x:9999?mode=caller│
│  │   └── Slot B ffmpeg (listener)   │     HTTP port 8808 ───► │  └── Browser Source (overlay)   │
│  └── run/videos_interpolated/       │                         │      http://10.x:8808/visuals/  │
│                                     │                         │      overlay                    │
└─────────────────────────────────────┘                         └─────────────────────────────────┘
```

### What was already finished (v0.5.8-beta)

- Crowd Control: submission page, QR code, `/api/next` pop, SQLite queue, rate limiting, badword filter, generator client (fail-open)
- SRT watcher: A/B slot design, ffmpeg restart per slot, inotify watch, OBS WebSocket toggle

### What was missing / added (v0.9.0-beta)

- **Crowd queue overlay** — OBS browser source at `/visuals/overlay` with QR, counters, next prompts
- **Public status API** — `/visuals/api/status?limit=3` returns queue preview + aggregate counts (no IPs)
- **Enhanced submission page** — live stats, upcoming prompts, links to overlay/status/QR
- **SRT watcher hardening** — prefers the actual active `worqspace/config.yaml`, supports `VF_CONFIG_FILE`, watches both `close_write` and `moved_to`, preloads the newest existing clip on startup, and waits for file readiness before swapping live
- **CLI banner** — overlay + status URLs printed on crowd server startup
- **Docs + ops** — this file, env examples, systemd unit examples

---

### Setup Steps

#### 1. GPU Server — Visual FaQtory

```bash
# Clone / extract to /opt/visual-faqtory
cd /opt/visual-faqtory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Runtime tools needed by the SRT watcher
sudo apt-get install -y ffmpeg inotify-tools
# ffprobe is usually bundled with ffmpeg packages
# Also: pip install obsws-python  (if OBS autoswap desired)
```

#### 2. Configure Crowd Control

```bash
cp vf-crowd-control.env.example vf-crowd-control.env
# Edit vf-crowd-control.env:
#   VF_CROWD_TOKEN=<long random string>
#   VF_CROWD_PUBLIC_URL=https://wonq.tv/visuals   (or your public URL)
```

Start manually:
```bash
source .venv/bin/activate
python vfaq_cli.py crowd --token "$VF_CROWD_TOKEN"
```

Or via systemd:
```bash
sudo cp ops/systemd/vf-crowd-control.service.example /etc/systemd/system/vf-crowd-control.service
# Edit paths in the unit if not at /opt/visual-faqtory
sudo systemctl daemon-reload
sudo systemctl enable --now vf-crowd-control
```

#### 3. Configure SRT Watcher

```bash
cp vf-obs-watcher-srt-endpoints.env.example vf-obs-watcher-srt-endpoints.env
# Edit vf-obs-watcher-srt-endpoints.env:
#   SRT_BIND_IP=0.0.0.0
#   SRT_PUBLIC_IP=10.20.30.22          ← your ZeroTier IP
#   SRT_PORT_A=9998
#   SRT_PORT_B=9999
#   OBS_AUTOSWAP=1                     ← or 0 for manual switching
#   OBS_WS_HOST=10.20.30.12            ← OBS machine ZeroTier IP
#   OBS_WS_PORT=4455
#   OBS_WS_PASSWORD=YourPassword
```

Start manually:
```bash
# Validate config/tooling first
./vf-obs-watcher-srt-endpoints.sh --smoke-check

# Optional: print scriptable status only
./vf-obs-watcher-srt-endpoints.sh --status

# Then run for real
./vf-obs-watcher-srt-endpoints.sh
```

Or via systemd:
```bash
sudo cp ops/systemd/vf-srt-watcher.service.example /etc/systemd/system/vf-srt-watcher.service
# Edit the unit/env paths if your install is not /opt/visual-faqtory
sudo systemctl daemon-reload
sudo systemctl enable --now vf-srt-watcher
```

By default the watcher follows `worqspace/config.yaml`. To point it at another active
config file, set `VF_CONFIG_FILE=/path/to/worqspace/config-live.yaml` in the env file or
systemd unit. `VF_WATCH_DIR` still overrides everything when you want a fixed explicit
watch directory.

#### 4. OBS Machine — Add Sources

**Media Source A:**
- Input: `srt://10.20.30.22:9998?mode=caller&latency=20`
- Uncheck "Local File"
- Network Buffering: match SRT_LATENCY (20ms)
- Reconnect delay: 1s

**Media Source B:**
- Input: `srt://10.20.30.22:9999?mode=caller&latency=20`
- Same settings as A

**Browser Source (overlay):**
- URL: `http://10.20.30.22:8808/visuals/overlay`
- Width: 1920, Height: 1080 (or match your canvas)
- Custom CSS: (leave empty)
- Refresh when scene becomes active: checked
- The `?limit=N` query param controls how many upcoming prompts to show (default 3, max 10)

---

### Crowd Overlay URLs

| Route | Purpose |
|---|---|
| `GET /visuals/` | Submission page with live stats |
| `GET /visuals/overlay` | OBS browser source overlay |
| `GET /visuals/overlay?limit=5` | Overlay showing 5 upcoming prompts |
| `GET /visuals/api/status` | JSON: queue preview + counters |
| `GET /visuals/api/status?limit=10` | JSON: up to 10 upcoming prompts |
| `GET /visuals/api/health` | JSON: health check + queue length |
| `GET /visuals/qr.png` | QR code image |
| `POST /visuals/api/submit` | Submit a prompt (JSON body) |
| `GET /visuals/api/next?token=X` | Pop next prompt (token-protected) |

### SRT Endpoint URLs

These are printed by the watcher on startup:
```
srt://<SRT_PUBLIC_IP>:<SRT_PORT_A>?mode=caller&latency=<SRT_LATENCY>
srt://<SRT_PUBLIC_IP>:<SRT_PORT_B>?mode=caller&latency=<SRT_LATENCY>
```

The watcher now also prints the resolved config file and watch directory. `WATCH_MODE=videos` means it is following `<output_dir>/videos`; `WATCH_MODE=videos_interpolated` means `finalizer.per_cycle_interpolation=true` pushed it to `<output_dir>/videos_interpolated`; `WATCH_MODE=override` means `VF_WATCH_DIR` won. New videos are picked up whether they are written in place or atomically moved into the watched folder. On startup it preloads the newest existing watched `.mp4` into both A/B playout slots so the endpoints are immediately useful before the next generation finishes.

---

### Best Settings for Seamless A/B Switch

The goal is zero-glitch when OBS switches from slot A to slot B (or vice versa).

**SRT Latency:** Keep `SRT_LATENCY=20` (20ms). This is the minimum practical
latency. Going lower risks packet loss; higher adds visible delay. For ZeroTier
across the same datacenter, 20ms is fine. If crossing continents, try 50-100.

**Warmup Time:** `WARMUP_SEC=1.2` is the sweet spot. This is how long the new
slot's ffmpeg runs before the old slot is disabled. The OBS SRT caller needs time
to connect, receive an I-frame, and start decoding. 1.2s covers this. If you see
a brief black flash on switch, increase to 2.0. If switching feels sluggish, try 0.8.

**Encoding for minimal switch artifacts:**
- `ENC_PRESET=veryfast` — fast enough for realtime, good enough quality
- GOP = 1 second (`OUT_FPS` frames). The watcher sets `-g $OUT_FPS -keyint_min $OUT_FPS`.
  Every second starts with an I-frame, so OBS can start decoding within 1s of connecting.
- `-tune zerolatency` — no B-frames, lowest possible encoding delay
- `-sc_threshold 0` — no scene-change I-frames (predictable GOP)
- `-muxdelay 0 -muxpreload 0` — no muxer buffering

**OBS Media Source settings for minimal latency:**
- Network Buffering: 20 MB (or match SRT latency)
- Reconnect Delay: 1s
- Use Hardware Decoding if available
- **Both sources overlapping on the same scene** — the watcher enables the new one
  first, waits WARMUP_SEC, then disables the old one. Both are briefly visible
  (the new one renders on top).

**If OBS_AUTOSWAP=0:** You manage switching manually via OBS. The SRT endpoints
still update and restart on new video, so you just toggle visibility yourself.

---

### Operator Checklist

1. Run `./vf-obs-watcher-srt-endpoints.sh --smoke-check` and confirm `CONFIG_FILE=`, `WATCH_DIR=`, `WATCH_MODE=`, `SRT_URL_A=`, and `SRT_URL_B=` are all correct.
2. If `WATCH_MODE=videos_interpolated`, your active Visual FaQtory config has `finalizer.per_cycle_interpolation=true`; otherwise it watches `<output_dir>/videos`.
3. From the OBS machine, test both printed SRT URLs with `ffplay` before opening OBS.
4. Leave `OBS_AUTOSWAP=0` for first boot if you want to verify the A/B streams manually before letting the watcher toggle scene items.
5. After the first real render lands, run `./vf-obs-watcher-srt-endpoints.sh --status` and confirm `ACTIVE_SLOT=` flips on each new clip.
6. Under systemd, use `journalctl -u vf-srt-watcher -f` during the first live run to confirm startup preload, warmup, and swap logs look sane.

### Smoke Tests

1. **Crowd Control server:**
   ```bash
   curl http://localhost:8808/visuals/api/health
   # {"ok":true,"queue_length":0,"version":"0.5.9-beta"}
   ```

2. **Submit a prompt:**
   ```bash
   curl -X POST http://localhost:8808/visuals/api/submit \
     -H "Content-Type: application/json" \
     -d '{"prompt":"neon jellyfish in space"}'
   ```

3. **Check status API:**
   ```bash
   curl http://localhost:8808/visuals/api/status?limit=3
   ```

4. **Pop a prompt:**
   ```bash
   curl "http://localhost:8808/visuals/api/next?token=YOUR_TOKEN"
   ```

5. **SRT connectivity:**
   ```bash
   # From the OBS machine, test with ffplay:
   ffplay "srt://10.20.30.22:9998?mode=caller&latency=20"
   ffplay "srt://10.20.30.22:9999?mode=caller&latency=20"
   ```

   Quick watcher status on the GPU box:
   ```bash
   ./vf-obs-watcher-srt-endpoints.sh --status
   ```

6. **Overlay in browser:**
   Open `http://10.20.30.22:8808/visuals/overlay` — should show QR, stats, queue.

---

### Troubleshooting

**SRT connection refused:** Check firewall / ZeroTier routing. Verify the ffmpeg
processes are running: `ps aux | grep ffmpeg`. Also verify `inotifywait` is installed
(`inotify-tools` package) and that `./vf-obs-watcher-srt-endpoints.sh --smoke-check` passes.

**OBS shows black:** The watcher now seeds both slots with the newest existing watched `.mp4` on
startup when available. If you still get black, run `./vf-obs-watcher-srt-endpoints.sh --status`
and verify `WATCH_DIR=` points at the actual active run folder.

**OBS autoswap not working:** Verify `obsws-python` is installed in the venv.
Check OBS_WS_HOST, OBS_WS_PORT, OBS_WS_PASSWORD. Try `OBS_AUTOSWAP=0` and
switch manually to isolate the issue.

**Overlay not updating:** Check browser source URL points to the correct
host:port. The overlay polls `/api/status` every 3 seconds.

**Queue shows stale data:** The submission page polls every 5 seconds, the
overlay every 3 seconds. If the server is down, stale data remains until the
next successful fetch.


---

## First Show / First Stream Checklist

1. Run `./vf-obs-watcher-srt-endpoints.sh --smoke-check` on the generator host.
2. Run `./vf-obs-watcher-srt-endpoints.sh --status --verbose` and confirm `WATCH_DIR_EXISTS=1` and the `SRT_URL_*` values are correct.
3. Validate a known clip with `./vf-obs-watcher-srt-endpoints.sh --validate-file /path/to/test.mp4`.
4. If you already have old clips in the watched folder, run `./vf-obs-watcher-srt-endpoints.sh --reseed-slots` before launching the service.
5. On the OBS machine, test both printed SRT URLs with `ffplay` before enabling autoswap.
6. For the first live show, consider `OBS_AUTOSWAP=0` until both endpoints prove stable.
7. Tail logs with `journalctl -u vf-srt-watcher -f` during first load-in and first live swap.

## Exact Manual Operator Drill

### 1) Empty watch dir startup

```bash
./vf-obs-watcher-srt-endpoints.sh --smoke-check
```
Expected runtime logs after real start:
- `No existing .mp4 clip found for startup preload`
- `Creating black placeholder for slot A`
- `Creating black placeholder for slot B`

### 2) Preload from existing clips

```bash
./vf-obs-watcher-srt-endpoints.sh --reseed-slots
```
Expected:
- `Re-seeded both slots from <filename>.mp4`
- `--status --verbose` shows both slot files present

### 3) Direct-write pickup

```bash
./ops/simulate-srt-ingest.sh --watch-dir ./run/videos --source /path/to/test.mp4 --mode direct-write
```
Expected watcher logs:
- `Loading completed file (...)`
- `Prewarming ...`
- `Switched to slot A` or `Switched to slot B`

### 4) Atomic-move pickup

```bash
./ops/simulate-srt-ingest.sh --watch-dir ./run/videos --source /path/to/test.mp4 --mode atomic-move
```
Expected watcher logs:
- `Loading completed file (MOVED_TO)` or similar
- the active slot flips

### 5) Repeated swaps

Repeat the helper call multiple times with different filenames. Expected: A → B → A → B slot alternation.

### 6) OBS autoswap ON

Set `OBS_AUTOSWAP=1` and valid OBS WebSocket settings. Expected startup log: `OBS autoswap: ENABLED`.

### 7) OBS autoswap OFF

Set `OBS_AUTOSWAP=0`. Expected startup log: `OBS autoswap: DISABLED — SRT endpoints active, manual switching required`.

### 8) Failure / restart recovery

- Delete one slot file while the watcher runs: the health monitor should recover it.
- Kill one ffmpeg process: the health monitor should log `ffmpeg slot X is not running; restarting`.
- Rename or unmount the watched dir: the watcher should recreate the folder and restart the inotify loop.
- Restart the systemd service: it should pass `--smoke-check`, restore slot files, and restart both SRT endpoints.

## Known-good watch-dir logic

- `VF_WATCH_DIR` set: watcher uses that exact path and reports `WATCH_MODE=override`.
- otherwise it prefers `worqspace/config.yaml` and only falls back to `worqspace/config-live.yaml` when `config.yaml` is absent.
- `finalizer.per_cycle_interpolation=false`: watcher uses `<run.output_dir>/videos`.
- `finalizer.per_cycle_interpolation=true`: watcher uses `<run.output_dir>/videos_interpolated`.

## Extra Operator Commands

```bash
./vf-obs-watcher-srt-endpoints.sh --status --verbose
./vf-obs-watcher-srt-endpoints.sh --validate-file /path/to/test.mp4
./vf-obs-watcher-srt-endpoints.sh --reseed-slots
```
