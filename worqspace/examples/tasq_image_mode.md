---
# ═══════════════════════════════════════════════════════════════════════════════
# TASQ.MD - IMAGE MODE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Image mode starts from a base image (JPG/PNG) and transforms it.
# The prompt guides HOW the image should be transformed/stylized.
# Useful for: Brand consistency, starting from concept art, photo stylization
#
# The base image provides:
#   - Initial composition
#   - Color palette reference  
#   - Subject matter
#
# The prompt adds:
#   - Style transformation
#   - Mood/atmosphere
#   - Visual effects
# ═══════════════════════════════════════════════════════════════════════════════

mode: image
base_image: ./seed_image.png
seed: 42
drift_preset: ambient

# Control how much the image changes (0.0-1.0)
# Lower = more faithful to original
# Higher = more creative freedom
denoise_strength: 0.45
---

# Visual Prompt

Transform into ethereal dreamscape, add soft particle effects,
subtle lens flares, increase atmospheric depth, volumetric lighting,
maintain original composition but evolve towards surrealist style,
soft gradients, gentle color shifts towards purple and teal tones

## Negative

drastic scene change, completely different subject,
harsh contrasts, stark changes, unrecognizable result,
text, watermark, blurry, low quality
