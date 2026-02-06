# Example: Prompt Bundle Configuration
# ═══════════════════════════════════════════════════════════════════════════════
#
# A complete prompt bundle for Visual FaQtory v0.0.7-alpha consists of:
#
#   worqspace/
#     tasq.md              ← Base creative prompt (REQUIRED)
#     negative_prompt.md   ← What to avoid (optional)
#     style_hints.md       ← Style + evolution constraints (optional)
#     motion_prompt.md     ← Video motion intent (optional)
#     config.yaml          ← Mechanical parameters (REQUIRED)
#
# HOW IT WORKS:
#
# 1. InstruQtor loads all 4 files via the PromptBundle loader
# 2. All file contents are passed to the LLM for context-aware refinement
# 3. The LLM returns: refined_prompt, motion_hint, video_prompt, etc.
# 4. Everything is stored in the briq JSON for full auditability
#
# NEGATIVE PROMPT PRECEDENCE (highest first):
#   1. tasq.md frontmatter `negative_prompt:` 
#   2. negative_prompt.md file
#   3. ## Negative section inside tasq.md body
#   4. config.yaml prompt_drift.negative_prompt
#
# SPLIT BACKENDS (v0.0.7):
#   You can use different backends for image and video generation:
#
#   backends:
#     image:
#       type: comfyui
#       api_url: http://localhost:8188
#     video:
#       type: replicate
#       api_token: ${REPLICATE_API_TOKEN}
#
#   If 'backends:' is present, it overrides the legacy 'backend:' config.
#   If only 'backends.image' is set, video inherits from image.
#
# ═══════════════════════════════════════════════════════════════════════════════
