  Visual FaQtory v0.9.3-beta
  ═══════════════════════════════════════
  Reinject Default ON | Hybrid split routing | ComfyUI + Venice + Veo

09:04:28 [INFO] [FaQtory] Starting run — mode: text | reinject: True | id: run_20260725_090428_5a5562
09:04:28 [INFO] [FaQtory] Backend: split(image=venice, video=venice, morph=venice)
09:04:28 [INFO] [SlidingStory] Starting story run. Reinject: True. Story: story.txt
09:04:28 [INFO] [SlidingStory] Loaded: motion_prompt.md (327 chars)
09:04:28 [INFO] [SlidingStory] Parsed 34 paragraphs → 35 cycles with window size ≤ 2
09:04:28 [INFO] [SlidingStory/SmartReinject] Async prefetch enabled.
09:04:28 [INFO] [Timing] authority=duration | fps=24, frames=None, duration=4.0
09:04:28 [INFO] [Timing] resolved: fps=24, frames=96, duration=4.000
09:04:28 [INFO] [SlidingStory] Cycle 1/35 — window paragraphs [1]
09:04:28 [INFO] [SlidingStory/Venice] Cycle 1: text_to_video
09:04:28 [WARNING] [Venice] Retrying /video/queue without optional field(s): audio, resolution
09:04:28 [INFO] [Venice] Job queued — id=019f9816-f1b… model=gemini-omni-flash-text-to-video op=text2vid duration=4s
09:05:01 [INFO] [Venice] text2vid complete — 31.4s, 6 poll(s), 1075 KB → video_001_venice.mp4
09:05:01 [INFO] [SlidingStory/Venice] Generated video → /x/visual-faqtory/run/videos/video_001.mp4
09:05:01 [INFO] [SlidingStory] Extract last frame by exact frame index: ffmpeg -y -i /x/visual-faqtory/run/videos/video_001.mp4 -vf select=eq(n\,95) -vsync 0 -frames:v 1 -q:v 2 /x/visual-faqtory/run/frames/lastframe_001.tmp.png
09:05:01 [INFO] [SlidingStory] Cycle 2/35 — window paragraphs [1, 2]
09:05:01 [INFO] [SlidingStory/Venice] Cycle 2: image_to_video from last frame
09:05:01 [INFO] [SlidingStory/SmartReinject] Scheduled async prefetch smart_reinject_002_for_003 (lastframe_001.png -> cycle 3)
  ⠴ Venice img2img — EDITING — 0.6s                                            09:05:02 [WARNING] [Venice] Retrying /video/queue without optional field(s): audio, resolution
09:05:02 [ERROR] [Venice] Image generation failed: Venice API error Invalid model id (HTTP 400): {"error": "Invalid model id"}
09:05:03 [INFO] [Venice] Job queued — id=019f9817-792… model=gemini-omni-flash-image-to-video op=img2vid duration=4s
  ⠋ Venice img2vid — PROCESSING — 19.2s poll#4                                 ^C09:05:22 [INFO] 
[FaQtory] Run interrupted by user
09:05:22 [INFO] 
Run interrupted by user (state saved for --resume)

getting this error now, please fix it and then run python3 vfaq_cli.py -n approach1 and let it run the full 34 paragraph story please.
