#!/usr/bin/env python3
"""
vfaq_cli.py - Visual FaQtory Command Line Interface
═══════════════════════════════════════════════════════════════════════════════

CLI for the QonQrete Visual FaQtory - automated long-form AI visual generation.

Supports real-time TURBO with audio reactivity, longcat stream mode for true autoregressive continuation, MIDI sidecar control and TouchDesigner integration via file watching and OSC.

Usage:
    python vfaq_cli.py run                         # Run with config.yaml settings
    python vfaq_cli.py run -n my-project           # Run as named project
    python vfaq_cli.py run -c 10                   # Run 10 cycles
    python vfaq_cli.py run --hours 2               # Run until 2 hours of content
    python vfaq_cli.py status                      # Check pipeline status
    python vfaq_cli.py backends                    # List available backends
    python vfaq_cli.py single                      # Run single test cycle
    python vfaq_cli.py assemble                    # Assemble videos into one
    python vfaq_cli.py assemble -n my-project      # Assemble project videos

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BANNER = """
 ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗         ███████╗ █████╗  ██████╗ ████████╗ ██████╗ ██████╗ ██╗   ██╗
 ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║         ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
 ██║   ██║██║███████╗██║   ██║███████║██║         █████╗  ███████║██║   ██║   ██║   ██║   ██║██████╔╝ ╚████╔╝
 ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║         ██╔══╝  ██╔══██║██║▄▄ ██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝
  ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗    ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║   ██║
   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝

  QonQrete Visual FaQtory v0.3.5-beta
  ═══════════════════════════════════════
"""


def _resolve_project_dir(args):
    """Resolve the output directory based on project name or default."""
    worqspace = Path(args.worqspace).resolve()
    if hasattr(args, 'name') and args.name:
        return worqspace / "qonstructions" / args.name
    return Path(args.output).resolve()


def cmd_run(args):
    """Run the visual generation pipeline."""
    from vfaq import VisualFaQtory

    worqspace = Path(args.worqspace).resolve()

    if not worqspace.exists():
        logger.error(f"Worqspace not found: {worqspace}")
        sys.exit(1)

    config_override = {}
    if args.backend:
        config_override['backend'] = {'type': args.backend}
    if hasattr(args, 'delay') and args.delay is not None:
        if 'cycle' not in config_override:
            config_override['cycle'] = {}
        config_override['cycle']['delay_seconds'] = args.delay
    # Audio reactivity CLI overrides (v0.1.0)
    if hasattr(args, 'bpm') and args.bpm is not None:
        if 'audio_reactivity' not in config_override:
            config_override['audio_reactivity'] = {}
        config_override['audio_reactivity']['bpm_manual'] = args.bpm
        config_override['audio_reactivity']['enabled'] = True
    if hasattr(args, 'no_audio_react') and args.no_audio_react:
        if 'audio_reactivity' not in config_override:
            config_override['audio_reactivity'] = {}
        config_override['audio_reactivity']['enabled'] = False
    if hasattr(args, 'base_pick') and args.base_pick:
        if 'inputs' not in config_override:
            config_override['inputs'] = {}
        if 'base_folders' not in config_override['inputs']:
            config_override['inputs']['base_folders'] = {}
        config_override['inputs']['base_folders']['pick_mode'] = args.base_pick

    # Auto-duration CLI overrides (v0.1.2)
    if hasattr(args, 'match_audio') and args.match_audio:
        if 'duration' not in config_override:
            config_override['duration'] = {}
        config_override['duration']['match_audio'] = True
    if hasattr(args, 'duration') and args.duration is not None:
        if 'duration' not in config_override:
            config_override['duration'] = {}
        config_override['duration']['mode'] = 'fixed'
        config_override['duration']['seconds'] = args.duration
    # Stream mode CLI override (v0.2.0)
    if hasattr(args, 'stream') and args.stream:
        # Enable both new and legacy stream configs to maximise compatibility
        if 'stream' not in config_override:
            config_override['stream'] = {}
        config_override['stream']['enabled'] = True
        # Also enable legacy stream_mode for backward compatibility
        if 'stream_mode' not in config_override:
            config_override['stream_mode'] = {}
        config_override['stream_mode']['enabled'] = True

    print(BANNER)

    project_name = args.name if hasattr(args, 'name') else None
    output_dir = Path(args.output).resolve()

    faqtory = VisualFaQtory(
        worqspace_dir=worqspace,
        output_dir=output_dir,
        config_override=config_override,
        project_name=project_name
    )

    cycles = args.cycles
    hours = args.hours

    if hours > 0:
        logger.info(f"Target duration: {hours} hours")
    elif cycles > 0:
        logger.info(f"Target cycles: {cycles}")
    else:
        logger.info("Running until stopped (Ctrl+C)")

    briqs = faqtory.run(
        cycles=cycles,
        target_hours=hours,
        resume=not args.fresh
    )

    logger.info(f"Completed {len(briqs)} cycles")


def cmd_single(args):
    """Run a single test cycle."""
    from vfaq import VisualFaQtory

    worqspace = Path(args.worqspace).resolve()
    output = Path(args.output).resolve()

    config_override = {}
    if args.backend:
        config_override['backend'] = {'type': args.backend}

    project_name = args.name if hasattr(args, 'name') else None

    faqtory = VisualFaQtory(
        worqspace_dir=worqspace,
        output_dir=output,
        config_override=config_override,
        project_name=project_name
    )

    logger.info("Running single test cycle...")
    briq = faqtory.run_single_cycle(cycle_index=args.cycle)

    logger.info(f"Cycle complete!")
    logger.info(f"  Briq ID: {briq.briq_id}")
    logger.info(f"  Status: {briq.status.value}")
    logger.info(f"  Looped video: {briq.looped_video_path}")
    logger.info(f"  Evolution: {briq.evolution_suggestion}")


def cmd_status(args):
    """Show pipeline status."""
    from vfaq import VisualFaQtory

    worqspace = Path(args.worqspace).resolve()
    output = Path(args.output).resolve()
    project_name = args.name if hasattr(args, 'name') else None

    faqtory = VisualFaQtory(
        worqspace_dir=worqspace,
        output_dir=output,
        project_name=project_name
    )

    status = faqtory.status()

    print(f"\n=== Visual FaQtory v0.3.5-beta Status ===")
    for key, value in status.items():
        print(f"  {key}: {value}")

    outputs = faqtory.list_outputs()
    print(f"\n=== Generated Videos ({len(outputs)}) ===")
    for vid in outputs[-10:]:
        print(f"  {vid.name}")
    if len(outputs) > 10:
        print(f"  ... and {len(outputs) - 10} more")


def cmd_backends(args):
    """List available backends."""
    from vfaq import list_available_backends

    print(f"\n=== Available Backends (v0.3.5-beta) ===\n")

    results = list_available_backends()
    for name, (available, message) in results.items():
        status = "✓" if available else "✗"
        print(f"  [{status}] {name:12} - {message}")

    print("\nTo use a backend, set in worqspace/config.yaml:")
    print("  backend:")
    print("    type: comfyui")
    print("    api_url: http://localhost:8188")


def cmd_assemble(args):
    """Assemble generated videos into one."""
    from vfaq.finalizer import Finalizer

    project_dir = _resolve_project_dir(args)
    videos_dir = project_dir / "videos"

    # Also check root of project dir for legacy layouts
    if videos_dir.exists():
        videos = sorted(videos_dir.glob("cycle*_video.mp4"))
    else:
        videos = sorted(project_dir.glob("cycle*_video.mp4"))

    if not videos:
        logger.error("No videos found to assemble")
        sys.exit(1)

    logger.info(f"Found {len(videos)} videos to assemble")

    if args.preview:
        videos = videos[:args.preview_count]
        logger.info(f"Preview mode: using first {len(videos)} videos")

    # Load config for finalizer settings
    worqspace = Path(args.worqspace).resolve()
    config_path = worqspace / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        config = yaml.safe_load(config_path.read_text()) or {}

    finalizer_config = config.get('finalizer', {})
    codec = config.get('looping', {}).get('output_codec', 'h264_nvenc')
    quality = config.get('looping', {}).get('output_quality', 18)

    finalizer = Finalizer(
        project_dir=project_dir,
        preferred_codec=codec,
        output_quality=quality,
        finalizer_config=finalizer_config
    )
    try:
        final_path = finalizer.finalize(cycle_video_paths=videos)
        logger.info(f"Assembly complete: {final_path}")

        # Run post-stitch finalizer if enabled
        deliverable = finalizer.run_post_stitch_finalizer()
        if deliverable:
            logger.info(f"Final deliverable: {deliverable}")

    except Exception as e:
        logger.error(f"Assembly failed: {e}")
        sys.exit(1)


def cmd_clean(args):
    """Clean output directory."""
    import shutil

    project_dir = _resolve_project_dir(args)

    if args.all:
        if project_dir.exists():
            shutil.rmtree(project_dir)
            project_dir.mkdir()
            logger.info(f"Cleaned all: {project_dir}")
    else:
        state_file = project_dir / "factory_state.json"
        if state_file.exists():
            state_file.unlink()
            logger.info("Removed state file (will start fresh)")


def cmd_live(args):
    """Run TURBO live mode with optional crowd queue."""
    import yaml

    worqspace = Path(args.worqspace).resolve()

    if not worqspace.exists():
        logger.error(f"Worqspace not found: {worqspace}")
        sys.exit(1)

    # Load config
    config_path = worqspace / "config.yaml"
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}

    # CLI overrides
    if args.fps:
        config.setdefault('turbo', {})['fps_target'] = args.fps
    if args.size:
        try:
            w, h = args.size.split('x')
            config.setdefault('turbo', {})['width'] = int(w)
            config.setdefault('turbo', {})['height'] = int(h)
        except ValueError:
            logger.error(f"Invalid size format: {args.size} (use WxH)")
            sys.exit(1)
    if args.output:
        config.setdefault('turbo', {})['output_path'] = args.output

    # Ensure turbo enabled
    config.setdefault('turbo', {})['enabled'] = True

    # Crowd setup
    crowd_queue = None
    crowd_server = None

    if args.crowd:
        config.setdefault('crowd', {})['enabled'] = True
        if args.crowd_port:
            config.setdefault('crowd', {}).setdefault('server', {})['port'] = args.crowd_port
        if args.crowd_public_url:
            config.setdefault('crowd', {}).setdefault('server', {})['public_base_url'] = args.crowd_public_url
        if args.crowd_token:
            config.setdefault('crowd', {}).setdefault('server', {}).setdefault('auth', {})['enabled'] = True
            config['crowd']['server']['auth']['token'] = args.crowd_token

        try:
            from vfaq.crowd_queue import PromptQueue
            from vfaq.crowd_server import CrowdPromptServer

            crowd_queue = PromptQueue(config.get('crowd', {}))
            crowd_server = CrowdPromptServer(crowd_queue, config.get('crowd', {}))
            url = crowd_server.start()
            logger.info(f"Crowd server started: {url}")
        except Exception as e:
            logger.warning(f"Crowd server failed to start: {e}")
            logger.warning("TURBO will continue without crowd input")

    # Start TURBO
    print(BANNER)
    print("  🚀 TURBO LIVE MODE")
    print()

    try:
        from vfaq.turbo_engine import TurboEngine

        engine = TurboEngine(
            config=config,
            worqspace_dir=worqspace,
            crowd_queue=crowd_queue,
        )
        engine.run_live()
    except KeyboardInterrupt:
        logger.info("TURBO stopped by user")
    except Exception as e:
        logger.error(f"TURBO failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="QonQrete Visual FaQtory v0.3.5-beta - Automated AI Visual Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vfaq_cli.py run                         # Run with defaults
  python vfaq_cli.py run -n cyberpunk-set        # Named project
  python vfaq_cli.py run -c 100                  # Run 100 cycles
  python vfaq_cli.py run --hours 2               # Generate 2 hours
  python vfaq_cli.py run -b mock                 # Use mock backend
  python vfaq_cli.py single                      # Test single cycle
  python vfaq_cli.py status -n cyberpunk-set     # Check project status
  python vfaq_cli.py backends                    # List backends
  python vfaq_cli.py assemble -n cyberpunk-set   # Combine project videos
        """
    )

    parser.add_argument('-w', '--worqspace', default='./worqspace',
                       help='Worqspace directory (default: ./worqspace)')
    parser.add_argument('-o', '--output', default='./qodeyard',
                       help='Output directory for unnamed runs (default: ./qodeyard)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run visual generation')
    run_parser.add_argument('-n', '--name',
                           help='Project name (stored in worqspace/qonstructions/<name>/)')
    run_parser.add_argument('-c', '--cycles', type=int, default=0,
                           help='Number of cycles (0 = use config/unlimited)')
    run_parser.add_argument('--hours', type=float, default=0,
                           help='Target duration in hours')
    run_parser.add_argument('-b', '--backend',
                           help='Override backend (mock/comfyui/diffusers/replicate)')
    run_parser.add_argument('-s', '--seed', type=int,
                           help='Override initial seed')
    run_parser.add_argument('--bpm', type=int, default=None,
                           help='Manual BPM override for audio sync')
    run_parser.add_argument('--audio', type=str, default=None,
                           help='Explicit audio file path (overrides base_audio)')
    run_parser.add_argument('--base-pick', type=str, default=None,
                           choices=['newest', 'oldest', 'random', 'alphabetical'],
                           help='Base folder pick mode override')
    run_parser.add_argument('--no-audio-react', action='store_true',
                           help='Disable audio reactivity even if enabled in config')
    run_parser.add_argument('--fresh', action='store_true',
                           help='Start fresh (ignore saved state)')
    run_parser.add_argument('--delay', type=float,
                           help='Delay between cycles in seconds (default: 5)')
    # Auto-duration flags (v0.1.2)
    run_parser.add_argument('--match-audio', action='store_true',
                           help='Match visual duration to audio length')
    run_parser.add_argument('--duration', type=float, default=None,
                           help='Fixed duration in seconds (overrides cycle count)')
    # Stream mode flag (v0.3.0)
    run_parser.add_argument('--stream', action='store_true',
                           help='Enable longcat stream continuation mode (autoregressive)')
    run_parser.set_defaults(func=cmd_run)

    # Single command
    single_parser = subparsers.add_parser('single', help='Run single test cycle')
    single_parser.add_argument('-n', '--name', help='Project name')
    single_parser.add_argument('--cycle', type=int, default=0,
                              help='Cycle index to run')
    single_parser.add_argument('-b', '--backend',
                              help='Override backend')
    single_parser.set_defaults(func=cmd_single)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.add_argument('-n', '--name', help='Project name')
    status_parser.set_defaults(func=cmd_status)

    # Backends command
    backends_parser = subparsers.add_parser('backends', help='List available backends')
    backends_parser.set_defaults(func=cmd_backends)

    # Assemble command
    assemble_parser = subparsers.add_parser('assemble', help='Assemble videos')
    assemble_parser.add_argument('-n', '--name', help='Project name')
    assemble_parser.add_argument('--preview', action='store_true',
                                help='Create quick preview')
    assemble_parser.add_argument('--preview-count', type=int, default=10,
                                help='Number of videos in preview')
    assemble_parser.set_defaults(func=cmd_assemble)

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean output')
    clean_parser.add_argument('-n', '--name', help='Project name')
    clean_parser.add_argument('--all', action='store_true',
                             help='Remove everything (not just state)')
    clean_parser.set_defaults(func=cmd_clean)

    # Live (TURBO) command (v0.3.5-beta)
    live_parser = subparsers.add_parser('live', help='TURBO live mode + crowd queue')
    live_parser.add_argument('--turbo', action='store_true', default=True,
                            help='Enable TURBO frame generation (default)')
    live_parser.add_argument('--fps', type=int, default=None,
                            help='Target FPS (default: from config)')
    live_parser.add_argument('--size', type=str, default=None,
                            help='Resolution WxH (e.g., 768x432)')
    live_parser.add_argument('--output', type=str, default=None,
                            help='Output path for live frame')
    live_parser.add_argument('--crowd', action='store_true',
                            help='Enable crowd prompt server')
    live_parser.add_argument('--crowd-port', type=int, default=None,
                            help='Crowd server port (default: 7777)')
    live_parser.add_argument('--crowd-public-url', type=str, default=None,
                            help='Public URL for crowd QR/link')
    live_parser.add_argument('--crowd-token', type=str, default=None,
                            help='Auth token for crowd submissions')
    live_parser.set_defaults(func=cmd_live)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
