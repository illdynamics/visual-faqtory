from pathlib import Path

import pytest

import vfaq_cli


def test_root_help_shows_root_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        vfaq_cli.main(["-h"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "backends" in out
    assert "crowd" in out


def test_root_version_matches_version_file(capsys):
    expected = Path(vfaq_cli.__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
    with pytest.raises(SystemExit) as exc:
        vfaq_cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert expected in out


def test_global_option_before_subcommand_is_preserved(monkeypatch):
    captured = {}

    def fake_backends(args):
        captured["worqspace"] = args.worqspace
        captured["run_dir"] = args.run_dir

    monkeypatch.setattr(vfaq_cli, "cmd_backends", fake_backends)
    vfaq_cli.main(["-w", "./worqspace", "--run-dir", "./run", "backends"])
    assert captured == {"worqspace": "./worqspace", "run_dir": "./run"}


def test_no_subcommand_defaults_to_run(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["command"] = args.command
        captured["dry_run"] = args.dry_run
        captured["worqspace"] = args.worqspace

    monkeypatch.setattr(vfaq_cli, "cmd_run", fake_run)
    vfaq_cli.main(["--dry-run", "-w", "./worqspace"])
    assert captured == {"command": "run", "dry_run": True, "worqspace": "./worqspace"}
