wicked@dextronux:/x/visual-faqtory$ python3 vfaq_cli.py -n newsflash1
/home/wicked/.asdf/installs/python/3.11.8/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.4) doe
sn't match a supported version!
  warnings.warn(

 ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗         ███████╗ █████╗  ██████╗ ████████╗ ██████╗ ██████╗ ██╗   ██╗
 ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║         ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
 ██║   ██║██║███████╗██║   ██║███████║██║         █████╗  ███████║██║   ██║   ██║   ██║   ██║██████╔╝ ╚████╔╝
 ╚██ ██╔╝██║╚════██║██║   ██║██╔══██║██║         ██╔══╝  ██╔══██║██║▄▄ ██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝
  ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗    ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║   ██║
   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝

  Visual FaQtory v0.9.4-beta
  ═══════════════════════════════════════
  Reinject Default ON | Hybrid split routing | ComfyUI + Venice + Veo

02:04:00 [INFO] [FaQtory] Starting run — mode: text | reinject: True | id: run_20260803_020400_dc3380
02:04:00 [INFO] [FaQtory] Backend: split(image=venice, video=venice, morph=venice)
02:04:00 [INFO] [SlidingStory] Starting story run. Reinject: True. Story: story.txt
02:04:00 [INFO] [SlidingStory] Loaded: motion_prompt.md (327 chars)
02:04:00 [INFO] [SlidingStory] Parsed 34 paragraphs → 35 cycles with window size ≤ 2
02:04:00 [INFO] [SlidingStory/SmartReinject] Async prefetch enabled.
02:04:00 [INFO] [Timing] authority=duration | fps=24, frames=None, duration=4.0
02:04:00 [INFO] [Timing] resolved: fps=24, frames=96, duration=4.000
02:04:00 [INFO] [SlidingStory] Cycle 1/35 — window paragraphs [1]
02:04:00 [INFO] [SlidingStory/Venice] Cycle 1: text_to_video
02:04:00 [WARNING] [Venice] Retrying /video/queue without optional field(s): audio, resolution
02:04:00 [INFO] [Venice] Job queued — id=019fc4ef-3b8… model=gemini-omni-flash-text-to-video op=text2vid duration=4s
02:04:13 [ERROR] Venice poll loop crashed: Venice API error prompt: Could not generate a video with the given inputs. Please try again with different inputs. (HTTP 400): {"error": "prompt:
Could not generate a video with the given inputs. Please try again with different inputs."}
Traceback (most recent call last):
  File "/x/visual-faqtory/vfaq/venice_backend.py", line 807, in _run_video_job
    raise RuntimeError(self._format_error_payload(response.status_code, last_status))
RuntimeError: Venice API error prompt: Could not generate a video with the given inputs. Please try again with different inputs. (HTTP 400): {"error": "prompt: Could not generate a video wi
th the given inputs. Please try again with different inputs."}
02:04:13 [ERROR] [Venice] Video generation failed: Venice API error prompt: Could not generate a video with the given inputs. Please try again with different inputs. (HTTP 400): {"error": "
prompt: Could not generate a video with the given inputs. Please try again with different inputs."}
02:04:13 [ERROR] [FaQtory] Run failed at cycle ~1: [Venice] Video generation failed in cycle 1: Venice API error prompt: Could not generate a video with the given inputs. Please try again w
ith different inputs. (HTTP 400): {"error": "prompt: Could not generate a video with the given inputs. Please try again with different inputs."}
02:04:13 [ERROR] Run failed: [Venice] Video generation failed in cycle 1: Venice API error prompt: Could not generate a video with the given inputs. Please try again with different inputs.
(HTTP 400): {"error": "prompt: Could not generate a video with the given inputs. Please try again with different inputs."}
02:04:13 [INFO] State saved. Rerun with --resume to continue from last checkpoint.

getting this error now, please fix it and then re-run this vfaq run plz by executing python3 vfaq_cli.py -n newsflash1 --resume and let it run the full run please.
