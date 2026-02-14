#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# vfaq_story_setup.sh — Interactive Story & Prompt File Setup
# Visual FaQtory v0.5.6-beta
# ═══════════════════════════════════════════════════════════════════════════════
#
# Helps you set up worqspace/story.txt and supporting prompt files.
# No external dependencies (no yq needed).
#
# Usage: bash vfaq_story_setup.sh [worqspace_dir]
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

WORQSPACE="${1:-./worqspace}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Visual FaQtory v0.5.6-beta — Story Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Worqspace: $WORQSPACE"
echo ""

# Ensure worqspace exists
mkdir -p "$WORQSPACE"

# ── story.txt ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STORY.TXT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Paste your story below. Separate paragraphs with blank lines."
echo "  Each paragraph = one visual cycle window step."
echo "  Press Ctrl+D (EOF) when done."
echo ""

STORY_FILE="$WORQSPACE/story.txt"
if [ -f "$STORY_FILE" ]; then
    PARA_COUNT=$(grep -c '^$' "$STORY_FILE" 2>/dev/null || echo "0")
    echo "  [Existing story.txt found with ~$PARA_COUNT paragraph breaks]"
    read -rp "  Overwrite? [y/N]: " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[yY] ]]; then
        echo "  Keeping existing story.txt"
    else
        echo "  Enter new story (Ctrl+D to finish):"
        cat > "$STORY_FILE"
    fi
else
    echo "  Enter your story (Ctrl+D to finish):"
    cat > "$STORY_FILE"
fi

# Validate paragraphs
if [ -f "$STORY_FILE" ]; then
    # Count paragraphs (blocks separated by blank lines)
    PARA_COUNT=$(awk 'BEGIN{c=0; in_para=0} /^[[:space:]]*$/{if(in_para) c++; in_para=0; next} {in_para=1} END{if(in_para) c++; print c}' "$STORY_FILE")
    echo ""
    echo "  ✓ Detected $PARA_COUNT paragraph(s) in story.txt"
    if [ "$PARA_COUNT" -lt 2 ]; then
        echo "  ⚠ WARNING: Less than 2 paragraphs detected."
        echo "    The sliding window engine works best with 2+ paragraphs."
        echo "    Separate paragraphs with blank lines."
    fi
fi

# ── Optional prompt files ──────────────────────────────────────────────────
for PROMPT_FILE in motion_prompt.md style_hints.md evolution_lines.md negative_prompt.md; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $PROMPT_FILE (optional)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    FPATH="$WORQSPACE/$PROMPT_FILE"
    if [ -f "$FPATH" ]; then
        echo "  [Exists: $(head -1 "$FPATH")...]"
        read -rp "  Edit? [y/N]: " EDIT
        if [[ "$EDIT" =~ ^[yY] ]]; then
            echo "  Enter new content (Ctrl+D to finish):"
            cat > "$FPATH"
        fi
    else
        read -rp "  Create? [y/N]: " CREATE
        if [[ "$CREATE" =~ ^[yY] ]]; then
            echo "  Enter content (Ctrl+D to finish):"
            cat > "$FPATH"
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "  Run the pipeline: python vfaq_cli.py"
echo "═══════════════════════════════════════════════════════════════"
echo ""
