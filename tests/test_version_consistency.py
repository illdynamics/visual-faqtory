from pathlib import Path

from vfaq.version import __version__

ROOT = Path(__file__).resolve().parents[1]
FILES_TO_CHECK = [
    ROOT / "vfaq_cli.py",
    ROOT / "README.md",
    ROOT / "doc/DOCUMENTATION.md",
    ROOT / "doc/BACKEND-VALIDATION-REPORT.md",
    ROOT / "doc/LIVE-INTEGRATION-GUIDE.md",
    ROOT / "doc/SRT-LIVE-OPS-REPORT.md",
    ROOT / "vfaq/venice_backend.py",
    ROOT / "ops/systemd/vf-srt-watcher.service.example",
    ROOT / "ops/systemd/vf-crowd-control.service.example",
]
STALE_CURRENT_VERSION_LITERALS = {"v0.7.7-beta", "v0.7.8-beta", "v0.5.9-beta"}


def test_version_file_matches_shared_version():
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == __version__


def test_key_user_facing_files_do_not_contain_stale_current_version_literals():
    offenders = {}
    for path in FILES_TO_CHECK:
        text = path.read_text(encoding="utf-8")
        found = sorted(v for v in STALE_CURRENT_VERSION_LITERALS if v in text)
        if found:
            offenders[str(path.relative_to(ROOT))] = found
    assert offenders == {}
