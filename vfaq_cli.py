#!/usr/bin/env python3
"""
vfaq_cli.py — Visual FaQtory Command Line Interface
═══════════════════════════════════════════════════════════════════════════════

CLI for the Visual FaQtory — automated long-form AI visual generation.

Usage:
    python vfaq_cli.py                          # Run with config.yaml settings (reinject ON)
    python vfaq_cli.py -n my-project            # Named project
    python vfaq_cli.py --no-reinject            # Disable reinject
    python vfaq_cli.py --mode image             # Override input mode
    python vfaq_cli.py --dry-run                # Validate config without generation
    python vfaq_cli.py status                   # Check pipeline status
    python vfaq_cli.py backends                 # List available backends

Part of Visual FaQtory v0.9.0-beta
"""
import os
import sys
import argparse
import logging
from pathlib import Path

from vfaq.version import __version__ as APP_VERSION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

KNOWN_COMMANDS = {"run", "status", "backends", "crowd"}
ROOT_HELP_FLAGS = {"-h", "--help"}
ROOT_VERSION_FLAGS = {"-V", "--version"}


def normalize_argv(argv):
    """Default to the run subcommand without breaking root help/version/global args."""
    argv = list(argv)
    if not argv:
        return ["run"]
    if any(arg in KNOWN_COMMANDS for arg in argv):
        return argv
    if any(arg in ROOT_HELP_FLAGS for arg in argv):
        return argv
    if any(arg in ROOT_VERSION_FLAGS for arg in argv):
        return argv
    return ["run", *argv]


def build_parser():
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument('-w', '--worqspace', default='./worqspace',
                               help='Worqspace directory (default: ./worqspace)')
    common_parent.add_argument('--run-dir', default='./run',
                               help='Run output directory (default: ./run)')

    parser = argparse.ArgumentParser(
        description=f"Visual FaQtory {APP_VERSION} — Automated AI Visual Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parent],
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
  python vfaq_cli.py crowd --token MY_SECRET    # Start Crowd Control server
  python vfaq_cli.py crowd --token MY_SECRET --public-url http://192.168.1.50:8808/visuals
        """
    )
    parser.add_argument('-V', '--version', action='version',
                        version=f'Visual FaQtory {APP_VERSION}')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    run_parser = subparsers.add_parser('run', parents=[common_parent], help='Run visual generation (default)')
    _add_run_args(run_parser)

    status_parser = subparsers.add_parser('status', parents=[common_parent], help='Show pipeline status')
    status_parser.set_defaults(func=cmd_status)

    backends_parser = subparsers.add_parser('backends', parents=[common_parent], help='List available backends')
    backends_parser.set_defaults(func=cmd_backends)

    crowd_parser = subparsers.add_parser('crowd', parents=[common_parent], help='Start Crowd Control server')
    crowd_parser.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    crowd_parser.add_argument('--port', type=int, default=8808, help='Bind port (default: 8808)')
    crowd_parser.add_argument('--prefix', default='/visuals', help='URL prefix (default: /visuals)')
    crowd_parser.add_argument('--public-url', default='https://wonq.tv/visuals', help='Public URL for QR code (default: https://wonq.tv/visuals)')
    crowd_parser.add_argument('--db-path', default='worqspace/crowdcontrol.sqlite3', help='SQLite database path')
    crowd_parser.add_argument('--token', default=None, help='Bearer token for /api/next (or set VF_CROWD_TOKEN env)')
    crowd_parser.add_argument('--badwords', default='worqspace/badwords.txt', help='Bad words filter file')
    crowd_parser.add_argument('--max-chars', type=int, default=300, help='Max prompt length (default: 300)')
    crowd_parser.add_argument('--rate-limit', type=int, default=600, help='Rate limit in seconds per IP (default: 600)')
    crowd_parser.add_argument('--max-queue', type=int, default=100, help='Max queue length (default: 100)')
    crowd_parser.set_defaults(func=cmd_crowd)
    return parser


BANNER = f"""
 ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗         ███████╗ █████╗  ██████╗ ████████╗ ██████╗ ██████╗ ██╗   ██╗
 ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║         ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
 ██║   ██║██║███████╗██║   ██║███████║██║         █████╗  ███████║██║   ██║   ██║   ██║   ██║██████╔╝ ╚████╔╝
 ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║         ██╔══╝  ██╔══██║██║▄▄ ██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝
  ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗    ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║   ██║
   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝

  Visual FaQtory {APP_VERSION}
  ═══════════════════════════════════════
  Reinject Default ON | Hybrid split routing | ComfyUI + Venice + Veo
"""


def cmd_run(args):
    """Run the visual generation pipeline."""
    from vfaq.visual_faqtory import VisualFaQtory

    worqspace = Path(args.worqspace).resolve()
    if not worqspace.exists():
        logger.error(f"Worqspace not found: {worqspace}")
        sys.exit(1)

    # ── Config file override ─────────────────────────────────────────────────
    # If --config is given, copy it over worqspace/config.yaml so VisualFaQtory
    # picks it up. Guard against SameFileError when source == destination.
    if args.config:
        config_src = Path(args.config).resolve()
        if not config_src.exists():
            logger.error(f"Config file not found: {config_src}")
            sys.exit(1)
        config_dest = worqspace / "config.yaml"
        import shutil
        if config_src != config_dest.resolve():
            shutil.copy2(config_src, config_dest)
            logger.info(f"[CLI] Using config: {config_src} → {config_dest}")
        else:
            logger.info(f"[CLI] Config already at {config_dest}, no copy needed")

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
        resume=getattr(args, 'resume', False),
    )

    try:
        faqtory.run()
    except KeyboardInterrupt:
        logger.info("\nRun interrupted by user (state saved for --resume)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Run failed: {e}")
        logger.info("State saved. Rerun with --resume to continue from last checkpoint.")
        sys.exit(1)


def cmd_status(args):
    """Show pipeline status with resume readiness."""
    run_dir = Path(args.run_dir).resolve()

    if run_dir.exists():
        from vfaq.run_state import format_status_report
        print(f"\n=== Visual FaQtory {APP_VERSION} Status ===")
        print(format_status_report(run_dir))
    else:
        print(f"No run directory found at {run_dir}")

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
    from vfaq.backends import list_available_backends, create_backend, extract_backend_config, get_backend_type_for_capability

    print(f"\n=== Available Backends ({APP_VERSION}) ===\n")
    results = list_available_backends()
    for name, (available, message) in results.items():
        status = "✓" if available else "✗"
        print(f"  [{status}] {name:12} - {message}")

    # ── Config-aware LTX check ───────────────────────────────────────────────
    # If the active config has backend.type: ltx_video, run a real readiness
    # check against the configured settings and show the result.
    worqspace = Path(args.worqspace).resolve()
    config_path = worqspace / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text()) or {}
            bc = extract_backend_config(config)
            if get_backend_type_for_capability(bc, "video") == "ltx_video":
                try:
                    ltx_backend = create_backend(bc)
                    avail, msg = ltx_backend.check_availability()
                    status = "✓" if avail else "✗"
                    print(f"\n  === Active LTX Config Check ===")
                    print(f"  [{status}] {msg}")
                except Exception as e:
                    print(f"\n  [✗] Active LTX config error: {e}")
        except Exception:
            pass  # Config parse failure is not fatal for backends listing
    print("    # and set comfyui_workflow_t2v / comfyui_workflow_i2v")


def cmd_crowd(args):
    """Start the Crowd Control server."""
    import os

    # Token validation
    token = args.token or os.environ.get("VF_CROWD_TOKEN")
    allow_no_token = os.environ.get("VF_CROWD_ALLOW_NO_TOKEN", "").lower() in ("true", "1", "yes")

    if not token and not allow_no_token:
        print("\n[ERROR] --token is required for the /api/next endpoint.")
        print("  Set --token <your-secret> or VF_CROWD_TOKEN=<your-secret>")
        print("  For development only: VF_CROWD_ALLOW_NO_TOKEN=true skips this check")
        sys.exit(1)

    if not token:
        token = "DEV_NO_TOKEN"
        print("\n[WARN] Running without pop token — /api/next is unprotected!")
        print("       This is fine for local dev, NOT for production.\n")

    from vfaq.crowd_control.models import CrowdControlConfig
    from vfaq.crowd_control.server import create_crowd_app

    config = CrowdControlConfig(
        enabled=True,
        base_url=f"http://{args.host}:{args.port}{args.prefix}",
        pop_token=token,
        public_url=args.public_url,
        prefix=args.prefix,
        db_path=args.db_path,
        badwords_path=args.badwords,
        max_chars=args.max_chars,
        rate_limit_seconds=args.rate_limit,
        max_queue=args.max_queue,
    )

    app = create_crowd_app(config)

    # Print startup banner
    local_url = f"http://{args.host}:{args.port}{args.prefix}"
    print(BANNER)
    print("  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │       CROWD CONTROL SERVER {APP_VERSION:<25}│")
    print("  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  Submit page : {local_url}/")
    print(f"  │  QR code     : {local_url}/qr.png")
    print(f"  │  OBS overlay : {local_url}/overlay")
    print(f"  │  Status JSON : {local_url}/api/status")
    print(f"  │  Health      : {local_url}/api/health")
    print(f"  │  Public URL  : {config.public_url}")
    print(f"  │  DB path     : {config.db_path}")
    print("  ├──────────────────────────────────────────────────────────┤")
    print("  │  config.yaml keys for generator integration:            │")
    print("  │                                                         │")
    print("  │    crowd_control:                                       │")
    print("  │      enabled: true                                      │")
    print(f"  │      base_url: \"{config.base_url}\"")
    print(f"  │      pop_token: \"{token}\"")
    print("  └──────────────────────────────────────────────────────────┘")
    print()

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
    )


def main(argv=None):
    parser = build_parser()
    normalized_argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


def _add_run_args(parser):
    """Add run-specific arguments to parser."""
    parser.add_argument('-n', '--name',
                        help='Project name (saved to worqspace/saved-runs/<name>/)')

    # Resume from checkpoint
    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='Resume from last checkpoint (requires existing run/)')

    # Reinject control — default ON (note: -r is now --resume, reinject has no short flag)
    reinject_group = parser.add_mutually_exclusive_group()
    reinject_group.add_argument('--reinject', action='store_true', default=True,
                                help='Enable reinject mode (default: ON)')
    reinject_group.add_argument('--no-reinject', '-R', action='store_true', default=False,
                                help='Disable reinject mode')

    parser.add_argument('--mode', choices=['text', 'image', 'video'],
                        help='Override input mode')
    parser.add_argument('-b', '--backend',
                        help='Override root backend.type for single-backend runs (mock/comfyui/animatediff/venice/veo/ltx_video/qwen_image_python/qwen_python). Split image/video/morph routing stays config-driven.')
    parser.add_argument('-s', '--seed', type=int,
                        help='Override base seed')
    parser.add_argument('--config', type=str,
                        help='Config file to use (e.g. worqspace/config-ltx.yaml). Copies to worqspace/config.yaml before run.')
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
