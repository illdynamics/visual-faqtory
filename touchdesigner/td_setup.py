#!/usr/bin/env python3
"""
td_setup.py - TouchDesigner Network Builder for Visual FaQtory
═══════════════════════════════════════════════════════════════════════════════

Run this script inside TouchDesigner to build the complete Visual FaQtory
FX network. Drag this file into the network editor or execute it via the
textport:

    exec(open('touchdesigner/td_setup.py').read())

The script creates:
  - Movie File In TOP reading live_output/current_frame.jpg
  - Cache TOP for frame buffering
  - Level TOP and HSV Adjust TOP for color correction
  - Feedback TOP + Composite TOP for recursive trails
  - Displace TOP for noise-driven warping
  - Text TOP + Composite TOP for HUD overlay
  - Null TOP as final output
  - Audio Device In CHOP + Analyze CHOP for RMS
  - MIDI In CHOP for controller input
  - OSC In CHOP for Visual FaQtory data (optional)

All operators are wired together. The network continues running even if
the AI generation stalls — Movie File In shows the last good frame.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""

# This script is meant to be executed inside TouchDesigner's Python environment.
# Outside of TD it serves as documentation for the network structure.

import sys

def build_network():
    """Build the Visual FaQtory TouchDesigner network."""

    # Check if we're running inside TouchDesigner
    try:
        # In TD, 'op' and 'me' are globally available
        root = op('/')  # noqa: F821
        parent_op = me.parent()  # noqa: F821
    except NameError:
        print("[Visual FaQtory TD Setup] Not running inside TouchDesigner.")
        print("Drag this file into TD's network editor or run via textport.")
        return

    base = parent_op

    # ─── Helper ──────────────────────────────────────────────────────────
    def safe_create(op_type, name, **kwargs):
        """Create an operator, replacing if it exists."""
        existing = base.op(name)
        if existing:
            existing.destroy()
        new_op = base.create(op_type, name)
        for k, v in kwargs.items():
            try:
                new_op.par[k] = v
            except Exception:
                pass
        return new_op

    # ═════════════════════════════════════════════════════════════════════
    # TOP CHAIN (Video FX)
    # ═════════════════════════════════════════════════════════════════════

    # 1. Movie File In — reads the AI output frame
    movie = safe_create(moviefileinTOP, 'moviefilein1',  # noqa: F821
        file='live_output/current_frame.jpg',
        reloadpulse=True,
    )
    movie.nodeX = 0
    movie.nodeY = 0

    # 2. Cache — decouples file read timing from render
    cache = safe_create(cacheTOP, 'cache1')  # noqa: F821
    cache.nodeX = 0
    cache.nodeY = -200
    cache.inputConnectors[0].connect(movie)

    # 3. Level — brightness/contrast/gamma control
    level = safe_create(levelTOP, 'level1',  # noqa: F821
        opacity=1.0,
        brightness1=1.0,
        contrast=1.0,
        gamma1=1.0,
    )
    level.nodeX = 0
    level.nodeY = -400
    level.inputConnectors[0].connect(cache)

    # 4. HSV Adjust — hue/saturation/value
    hsv = safe_create(hsvadjustTOP, 'hsvadjust1',  # noqa: F821
        satmult=1.1,
    )
    hsv.nodeX = 0
    hsv.nodeY = -600
    hsv.inputConnectors[0].connect(level)

    # 5. Feedback — recursive trails
    feedback = safe_create(feedbackTOP, 'feedback1')  # noqa: F821
    feedback.nodeX = -300
    feedback.nodeY = -800

    # 6. Composite — blend current frame with feedback
    comp1 = safe_create(compositeTOP, 'composite1',  # noqa: F821
        operand='Add',
        opacity=0.3,
    )
    comp1.nodeX = 0
    comp1.nodeY = -800
    comp1.inputConnectors[0].connect(hsv)
    comp1.inputConnectors[1].connect(feedback)

    # Wire feedback to read from composite
    feedback.inputConnectors[0].connect(comp1)

    # 7. Noise — displacement source
    noise = safe_create(noiseTOP, 'noise1',  # noqa: F821
        type='sparse',
        monochrome=False,
        resolutionw=256,
        resolutionh=256,
    )
    noise.nodeX = -300
    noise.nodeY = -1000

    # 8. Displace — noise-driven warping
    displace = safe_create(displaceTOP, 'displace1',  # noqa: F821
        displacex=5.0,
        displacey=5.0,
    )
    displace.nodeX = 0
    displace.nodeY = -1000
    displace.inputConnectors[0].connect(comp1)
    displace.inputConnectors[1].connect(noise)

    # 9. Text — HUD overlay showing current prompt
    text = safe_create(textTOP, 'text1',  # noqa: F821
        text='Visual FaQtory v0.3.5-beta',
        fontsizex=18,
        alignx='left',
        aligny='bottom',
        bgcolorr=0, bgcolorg=0, bgcolorb=0,
        bgalpha=0.5,
    )
    text.nodeX = -300
    text.nodeY = -1200

    # 10. Composite — overlay text on video
    comp2 = safe_create(compositeTOP, 'composite2',  # noqa: F821
        operand='Over',
    )
    comp2.nodeX = 0
    comp2.nodeY = -1200
    comp2.inputConnectors[0].connect(displace)
    comp2.inputConnectors[1].connect(text)

    # 11. Null — final output
    null_out = safe_create(nullTOP, 'null_out')  # noqa: F821
    null_out.nodeX = 0
    null_out.nodeY = -1400
    null_out.inputConnectors[0].connect(comp2)

    # ═════════════════════════════════════════════════════════════════════
    # CHOP CHAIN (Audio + MIDI)
    # ═════════════════════════════════════════════════════════════════════

    # Audio Device In
    audio_in = safe_create(audiodeviceinCHOP, 'audiodevin1')  # noqa: F821
    audio_in.nodeX = 600
    audio_in.nodeY = 0

    # Analyze — RMS + peak
    analyze = safe_create(analyzeCHOP, 'analyze1',  # noqa: F821
        function='rms',
    )
    analyze.nodeX = 600
    analyze.nodeY = -200
    analyze.inputConnectors[0].connect(audio_in)

    # Audio Filter — kick band (60-180 Hz)
    audio_filter = safe_create(audiofilterCHOP, 'audiofilter1',  # noqa: F821
        filtertype='bandpass',
        cutofflog=120,
        bandwidth=2,
    )
    audio_filter.nodeX = 600
    audio_filter.nodeY = -400
    audio_filter.inputConnectors[0].connect(audio_in)

    # MIDI In
    midi_in = safe_create(midiinCHOP, 'midiin1')  # noqa: F821
    midi_in.nodeX = 900
    midi_in.nodeY = 0

    # OSC In (for Visual FaQtory data)
    osc_in = safe_create(oscinCHOP, 'oscin1',  # noqa: F821
        port=6000,
    )
    osc_in.nodeX = 900
    osc_in.nodeY = -200

    print("[Visual FaQtory] TouchDesigner network built successfully!")
    print("  → Movie File In reads: live_output/current_frame.jpg")
    print("  → Final output: null_out")
    print("  → Audio RMS: analyze1")
    print("  → MIDI: midiin1")
    print("  → OSC: oscin1 (port 6000)")


# Run if executed
if __name__ == '__main__':
    build_network()
else:
    # Also run when dragged into TD
    try:
        build_network()
    except Exception as e:
        print(f"[Visual FaQtory TD Setup] Error: {e}")
