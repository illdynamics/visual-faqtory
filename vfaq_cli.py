#!/usr/bin/env python3
"""
vfaq_cli.py — Visual FaQtory Command Line Interface
═══════════════════════════════════════════════════════════════════════════════

CLI for the QonQrete Visual FaQtory — automated long-form AI visual generation.

Usage:
    python vfaq_cli.py                          # Run with config.yaml settings (reinject ON)
    python vfaq_cli.py -n my-project            # Named project
    python vfaq_cli.py --no-reinject            # Disable reinject
    python vfaq_cli.py --mode image             # Override input mode
    python vfaq_cli.py --dry-run                # Validate config without generation
    python vfaq_cli.py status                   # Check pipeline status
    python vfaq_cli.py backends                 # List available backends

Part of QonQrete Visual FaQtory v0.5.6-beta
"""
import os
import sys
import argparse
import logging
from pathlib import Path

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

  QonQrete Visual FaQtory v0.5.6-beta
  ═══════════════════════════════════════
  Reinject Default ON | ComfyUI Backend | Paragraph Story Engine
"""


def cmd_run(args):
    """Run the visual generation pipeline."""
    from vfaq.visual_faqtory import VisualFaQtory

    worqspace = Path(args.worqspace).resolve()
    if not worqspace.exists():
        logger.error(f"Worqspace not found: {worqspace}")
        sys.exit(1)

    # Determine reinject state
    reinject = True  # Default ON
    if args.no_reinject:
        reinject = False

    config_override = {}
    if args.backend:
        config_override['backend'] = {'type': args.backend}
    if args.seed is not None:
        config_override.setdefault('paragraph_story', {})['seed_base'] = args.seed

    # LoRA CLI overrides
    lora_cfg = {}
    if args.lora_enabled:
        lora_cfg['enabled'] = True
    if args.no_lora:
        lora_cfg['enabled'] = False
    if args.lora_path:
        lora_cfg['path'] = args.lora_path
    if args.lora_strength is not None:
        lora_cfg['strength'] = args.lora_strength
    if lora_cfg:
        lora_cfg.setdefault('backend', 'comfyui')
        config_override['lora'] = lora_cfg

    print(BANNER)

    run_dir = Path(args.run_dir).resolve()

    faqtory = VisualFaQtory(
        worqspace_dir=worqspace,
        run_dir=run_dir,
        config_override=config_override,
        project_name=args.name,
        reinject=reinject,
        mode_override=args.mode,
        dry_run=args.dry_run,
    )

    try:
        faqtory.run()
    except KeyboardInterrupt:
        logger.info("\nRun interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Run failed: {e}")
        sys.exit(1)


def cmd_status(args):
    """Show pipeline status."""
    import json

    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / "faqtory_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"\n=== Visual FaQtory v0.5.6-beta Status ===")
        for key, value in state.items():
            print(f"  {key}: {value}")
    else:
        print(f"No active run state found at {state_path}")

    # Check saved runs
    worqspace = Path(args.worqspace).resolve()
    saved_runs = worqspace / "saved-runs"
    if saved_runs.exists():
        runs = sorted(saved_runs.iterdir())
        if runs:
            print(f"\n=== Saved Runs ({len(runs)}) ===")
            for r in runs:
                if r.is_dir():
                    print(f"  {r.name}")


def cmd_backends(args):
    """List available backends."""
    from vfaq.backends import list_available_backends

    print(f"\n=== Available Backends (v0.5.6-beta) ===\n")
    results = list_available_backends()
    for name, (available, message) in results.items():
        status = "✓" if available else "✗"
        print(f"  [{status}] {name:12} - {message}")

    print("\nTo use a backend, set in worqspace/config.yaml:")
    print("  backend:")
    print("    type: comfyui")
    print("    api_url: http://localhost:8188")


def main():
    parser = argparse.ArgumentParser(
        description="QonQrete Visual FaQtory v0.5.6-beta — Automated AI Visual Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vfaq_cli.py                            # Run with defaults (reinject ON)
  python vfaq_cli.py -n cyberpunk-set           # Named project
  python vfaq_cli.py --no-reinject              # Disable reinject
  python vfaq_cli.py -R                         # Disable reinject (short flag)
  python vfaq_cli.py --mode image               # Use image mode
  python vfaq_cli.py --mode video               # Use video mode
  python vfaq_cli.py --dry-run                  # Validate without generation
  python vfaq_cli.py -n test -b mock            # Mock backend test
  python vfaq_cli.py status                     # Check status
  python vfaq_cli.py backends                   # List backends
        """
    )

    # Global options
    parser.add_argument('-w', '--worqspace', default='./worqspace',
                        help='Worqspace directory (default: ./worqspace)')
    parser.add_argument('--run-dir', default='./run',
                        help='Run output directory (default: ./run)')
    parser.add_argument('-V', '--version', action='version',
                        version='Visual FaQtory v0.5.6-beta')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command (also default when no subcommand)
    run_parser = subparsers.add_parser('run', help='Run visual generation (default)')
    _add_run_args(run_parser)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.set_defaults(func=cmd_status)

    # Backends command
    backends_parser = subparsers.add_parser('backends', help='List available backends')
    backends_parser.set_defaults(func=cmd_backends)

    # Check if first non-flag arg is a known subcommand
    known_commands = {'run', 'status', 'backends'}
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands and not sys.argv[1].startswith('-'):
        # First arg is not a known command - insert 'run'
        sys.argv.insert(1, 'run')
    elif len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith('-')):
        # No args or first arg is a flag - insert 'run'
        sys.argv.insert(1, 'run')

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


def _add_run_args(parser):
    """Add run-specific arguments to parser."""
    parser.add_argument('-n', '--name',
                        help='Project name (saved to worqspace/saved-runs/<name>/)')

    # Reinject control — default ON
    reinject_group = parser.add_mutually_exclusive_group()
    reinject_group.add_argument('--reinject', '-r', action='store_true', default=True,
                                help='Enable reinject mode (default: ON)')
    reinject_group.add_argument('--no-reinject', '-R', action='store_true', default=False,
                                help='Disable reinject mode')

    parser.add_argument('--mode', choices=['text', 'image', 'video'],
                        help='Override input mode')
    parser.add_argument('-b', '--backend',
                        help='Override backend (mock/comfyui)')
    parser.add_argument('-s', '--seed', type=int,
                        help='Override base seed')
    parser.add_argument('--config', type=str,
                        help='Override config file path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and inputs without generation')

    # LoRA options
    parser.add_argument('--lora-enabled', action='store_true',
                        help='Enable LoRA injection')
    parser.add_argument('--no-lora', action='store_true',
                        help='Disable LoRA injection')
    parser.add_argument('--lora-path', type=str,
                        help='Path to LoRA safetensors file')
    parser.add_argument('--lora-strength', type=float,
                        help='LoRA weight (0.0 to 1.0)')

    parser.set_defaults(func=cmd_run)


if __name__ == '__main__':
    main()
