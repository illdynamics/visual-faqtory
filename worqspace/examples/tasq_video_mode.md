---
# ═══════════════════════════════════════════════════════════════════════════════
# TASQ.MD - VIDEO MODE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Video mode starts from a base video (MP4/MOV) and transforms it.
# The prompt guides the stylization while preserving motion/timing.
#
# Use cases:
#   - Stylize existing footage
#   - Apply consistent aesthetic to clips
#   - Transform real footage into AI art
#   - Loop existing content with style transfer
#
# The base video provides:
#   - Motion and timing
#   - Scene structure
#   - Subject tracking
#
# The prompt adds:
#   - Visual style
#   - Color grading
#   - Atmospheric effects
# ═══════════════════════════════════════════════════════════════════════════════

mode: video
base_video: ./seed_video.mp4
seed: 42

# Video-specific settings
# motion_bucket_id: 127  # Keep original motion (higher = more motion)
# noise_aug_strength: 0.02  # Slight variation between frames
---

# Visual Prompt

Stylize with neon cyberpunk aesthetics, add glowing edge detection,
color grade to deep blue and electric purple tones,
maintain all motion and timing, enhance atmospheric fog,
add subtle scan lines and digital artifacts,
retro-futuristic VHS aesthetic, synthwave color palette

## Negative

change subject, alter motion, different timing,
lose original content, text overlay, watermark,
completely different scene, static result
