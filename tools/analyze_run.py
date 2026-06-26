#!/usr/bin/env python3
"""
analyze_run.py — Run Artifact Forensic Analyzer (v0.9.3+)
═══════════════════════════════════════════════════════════════════════════════

Scans a run directory and prints a structured diagnostic report.

Usage:
  python tools/analyze_run.py [run_dir]
  python tools/analyze_run.py run/ --json
  python tools/analyze_run.py run/ --timeline
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def scan_briq(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "analyzed_at": datetime.now().isoformat(),
        "cycles": {},
        "crashes": [],
        "crowd_events": [],
        "anomalies": [],
        "swap_events": [],
        "summary": {},
    }

    briqs_dir = run_dir / "briqs"
    obs_dir = run_dir / "obs"
    diag_file = obs_dir / ".watcher_diagnostics.jsonl"
    state_path = run_dir / "faqtory_state.json"

    # Load state
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            report["faqtory_state"] = {
                "run_id": state.get("run_id"),
                "status": state.get("status"),
                "cycles_completed": state.get("cycles_completed"),
                "cycles_planned": state.get("cycles_planned"),
                "error": state.get("error_message"),
                "start": state.get("start_time"),
                "end": state.get("end_time"),
            }
        except Exception as e:
            report["anomalies"].append(f"Failed to parse state: {e}")

    # Scan briqs
    if briqs_dir.exists():
        for briq_path in sorted(briqs_dir.glob("cycle_*.json")):
            briq = scan_briq(briq_path)
            if briq is None:
                report["anomalies"].append(f"Unparseable: {briq_path.name}")
                continue
            ci = briq.get("cycle_index", 0)
            cycle_info = {
                "cycle": ci,
                "crowd_used": briq.get("crowd_control", {}).get("used", False),
                "crowd_prompt": briq.get("crowd_control", {}).get("prompt_preview"),
                "video": briq.get("paths", {}).get("video"),
                "crashed": briq.get("cycle_crashed", False),
                "crash_error": briq.get("crash_error"),
            }
            if briq.get("cycle_crashed"):
                report["crashes"].append({
                    "cycle": ci,
                    "error": briq.get("crash_error", "unknown"),
                    "traceback": briq.get("crash_traceback", "")[:500],
                })
            cc = briq.get("crowd_control", {})
            if cc.get("used") and cc.get("prompt_preview"):
                report["crowd_events"].append({
                    "cycle": ci,
                    "prompt_id": cc.get("prompt_id"),
                    "prompt": cc.get("prompt_preview"),
                    "acked": cc.get("acked"),
                    "requeued": cc.get("requeued_on_failure"),
                })
            report["cycles"][ci] = cycle_info

    # Check OBS diagnostics
    if diag_file.exists():
        try:
            for line in diag_file.read_text().strip().split('\n'):
                if not line.strip():
                    continue
                report["swap_events"].append(json.loads(line))
        except Exception as e:
            report["anomalies"].append(f"Failed to parse diagnostics: {e}")

    # Summary
    total = len(report["cycles"])
    crashed = len(report["crashes"])
    crowd = len(report["crowd_events"])
    swap = len(report["swap_events"])
    swap_fail = sum(1 for e in report["swap_events"]
                    if e.get("event") in ("play_failed", "fallback_triggered", "fallback_failed"))

    report["summary"] = {
        "total_briqs": total,
        "crashed_cycles": crashed,
        "crowd_prompt_cycles": crowd,
        "swap_events": swap,
        "swap_failures": swap_fail,
        "anomalies_count": len(report["anomalies"]),
        "health": "healthy" if crashed == 0 and swap_fail == 0
        else "degraded" if crashed <= 2 else "critical",
    }
    return report


def print_report(report: Dict, *, json_mode: bool = False, timeline: bool = False):
    if json_mode:
        print(json.dumps(report, indent=2, default=str))
        return

    s = report["summary"]
    state = report.get("faqtory_state", {})
    print("═" * 60)
    print("  VISUAL FAQTORY — Run Forensic Analyzer")
    print("═" * 60)
    if state:
        print(f"  Status:   {state.get('status','?')}  Cycles: {state.get('cycles_completed','?')}/{state.get('cycles_planned','?')}")
        print(f"  Error:    {state.get('error','(none)')}")
    print("─" * 60)
    print(f"  Briqs: {s['total_briqs']}  Crashes: {s['crashed_cycles']}  Crowd prompts: {s['crowd_prompt_cycles']}")
    print(f"  Swap events: {s['swap_events']}  Swap failures: {s['swap_failures']}  Anomalies: {s['anomalies_count']}")
    print(f"  Health: {s['health'].upper()}")
    print("═" * 60)

    if report["crashes"]:
        print("\n  CRASHES:")
        for c in report["crashes"]:
            print(f"    Cycle {c['cycle']}: {c['error'][:120]}")

    if report["anomalies"]:
        print("\n  ANOMALIES:")
        for a in report["anomalies"]:
            print(f"    • {a}")

    if report["crowd_events"]:
        print(f"\n  CROWD PROMPTS ({len(report['crowd_events'])}):")
        for ce in report["crowd_events"]:
            acked = "✓" if ce.get("acked") else "✗"
            print(f"    Cycle {ce['cycle']} [{acked}]: {str(ce.get('prompt','?'))[:80]}")

    if timeline and report["cycles"]:
        print(f"\n  CYCLE TIMELINE:")
        for ci in sorted(report["cycles"].keys()):
            c = report["cycles"][ci]
            cc = "🎤" if c.get("crowd_used") else "  "
            crash = "💥" if c.get("crashed") else "  "
            print(f"    {cc}{crash} Cycle {ci:03d}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Artifact Forensic Analyzer")
    parser.add_argument("run_dir", nargs="?", default="./run", help="Run directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--timeline", action="store_true", help="Show timeline")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    report = analyze_run(run_dir)
    print_report(report, json_mode=args.json, timeline=args.timeline)
    if report["summary"]["health"] == "critical":
        sys.exit(2)


if __name__ == "__main__":
    main()
