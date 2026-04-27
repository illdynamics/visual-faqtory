import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


class SrtWatcherTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix='vfaq_srt_'))
        self.worqspace_dir = self.temp_dir / 'worqspace'
        self.worqspace_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = self.temp_dir / 'vf-obs-watcher-srt-endpoints.sh'
        shutil.copy2(self.repo_root / 'vf-obs-watcher-srt-endpoints.sh', self.script_path)
        self.script_path.chmod(self.script_path.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_yaml(self, rel_path: str, data: dict) -> Path:
        path = self.temp_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        return path

    def _make_fake_tool(self, name: str, body: str = 'exit 0\n') -> Path:
        tool = self.temp_dir / 'bin' / name
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_text('#!/usr/bin/env bash\nset -e\n' + body, encoding='utf-8')
        tool.chmod(0o755)
        return tool

    def _run_script(self, *args, extra_env=None):
        env = os.environ.copy()
        env['VF_SRT_ENV'] = str(self.temp_dir / 'missing.env')
        if extra_env:
          env.update({k: str(v) for k, v in extra_env.items()})
        return subprocess.run(
            ['bash', str(self.script_path), *args],
            cwd=self.temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def _parse_key_value_lines(self, text: str) -> dict:
        result = {}
        for line in text.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                result[key.strip()] = value.strip()
        return result

    def test_status_prefers_config_yaml_and_uses_output_dir(self):
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': False}})
        self._write_yaml('worqspace/config-live.yaml', {'run': {'output_dir': './live-run'}, 'finalizer': {'per_cycle_interpolation': True}})
        proc = self._run_script('--status')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['CONFIG_FILE'], str(self.temp_dir / 'worqspace' / 'config.yaml'))
        self.assertEqual(status['WATCH_DIR'], str(self.temp_dir / 'custom-run' / 'videos'))

    def test_status_honors_vf_config_file_override(self):
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': False}})
        live_cfg = self._write_yaml('worqspace/config-live.yaml', {'run': {'output_dir': './live-run'}, 'finalizer': {'per_cycle_interpolation': True}})
        proc = self._run_script('--status', extra_env={'VF_CONFIG_FILE': live_cfg})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['CONFIG_FILE'], str(live_cfg))
        self.assertEqual(status['WATCH_DIR'], str(self.temp_dir / 'live-run' / 'videos_interpolated'))

    def test_status_reports_watch_mode_for_per_cycle_interpolation(self):
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': True}})
        proc = self._run_script('--status')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['WATCH_MODE'], 'videos_interpolated')
        self.assertEqual(status['WATCH_DIR'], str(self.temp_dir / 'custom-run' / 'videos_interpolated'))

    def test_status_reports_newest_existing_mp4_for_startup_preload(self):
        watch_dir = self.temp_dir / 'run' / 'videos'
        watch_dir.mkdir(parents=True, exist_ok=True)
        old_file = watch_dir / 'older.mp4'
        new_file = watch_dir / 'newer.mp4'
        old_file.write_bytes(b'old')
        new_file.write_bytes(b'new')
        os.utime(old_file, (1_700_000_000, 1_700_000_000))
        os.utime(new_file, (1_800_000_000, 1_800_000_000))
        proc = self._run_script('--status', extra_env={'VF_WATCH_DIR': watch_dir})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['PRELOAD_FILE'], str(new_file))

    def test_status_verbose_reports_slot_and_pid_details(self):
        playdir = self.temp_dir / 'run' / 'obs'
        playdir.mkdir(parents=True, exist_ok=True)
        (playdir / 'current_A.mp4').write_bytes(b'A')
        (playdir / 'current_B.mp4').write_bytes(b'B')
        proc = self._run_script('--status', '--verbose', extra_env={'VF_PLAYOUT_DIR': playdir, 'VF_FFPROBE_BIN': str(self.temp_dir / 'bin' / 'missing-ffprobe')})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['SLOT_A_PRESENT'], '1')
        self.assertEqual(status['SLOT_B_PRESENT'], '1')
        self.assertEqual(status['SLOT_A_VALID'], '1')
        self.assertEqual(status['SLOT_B_VALID'], '1')


    def test_status_verbose_reports_ffprobe_available_when_present(self):
        playdir = self.temp_dir / 'run' / 'obs'
        playdir.mkdir(parents=True, exist_ok=True)
        (playdir / 'current_A.mp4').write_bytes(b'A')
        ffprobe = self._make_fake_tool('ffprobe', body='echo video\n')
        proc = self._run_script('--status', '--verbose', extra_env={'VF_PLAYOUT_DIR': playdir, 'VF_FFPROBE_BIN': ffprobe, 'OBS_AUTOSWAP': '0'})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['FFPROBE_AVAILABLE'], '1')

    def test_status_verbose_reports_ffprobe_unavailable_when_missing(self):
        playdir = self.temp_dir / 'run' / 'obs'
        playdir.mkdir(parents=True, exist_ok=True)
        (playdir / 'current_A.mp4').write_bytes(b'A')
        proc = self._run_script('--status', '--verbose', extra_env={'VF_PLAYOUT_DIR': playdir, 'VF_FFPROBE_BIN': str(self.temp_dir / 'bin' / 'missing-ffprobe'), 'OBS_AUTOSWAP': '0'})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        status = self._parse_key_value_lines(proc.stdout)
        self.assertEqual(status['FFPROBE_AVAILABLE'], '0')

    def test_status_is_quiet_and_does_not_emit_cleanup_logs(self):
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': False}})
        proc = self._run_script('--status')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn('Shutting down', proc.stdout)
        self.assertNotIn('Cleanup complete', proc.stdout)

    def test_smoke_check_returns_nonzero_when_inotifywait_missing(self):
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': False}})
        proc = self._run_script('--smoke-check', extra_env={'OBS_AUTOSWAP': '0', 'VF_INOTIFYWAIT_BIN': str(self.temp_dir / 'bin' / 'missing-inotifywait')})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('inotifywait required', proc.stderr)

    def test_process_file_swaps_active_slot_without_obs_autoswap(self):
        playdir = self.temp_dir / 'run' / 'obs'
        playdir.mkdir(parents=True, exist_ok=True)
        (playdir / 'current_A.mp4').write_bytes(b'A')
        (playdir / 'current_B.mp4').write_bytes(b'B')
        source_file = self.temp_dir / 'incoming.mp4'
        source_file.write_bytes(b'new clip data')
        inotifywait = self._make_fake_tool('inotifywait')
        ffmpeg = self._make_fake_tool('ffmpeg')
        proc = self._run_script('--process-file', str(source_file), extra_env={'OBS_AUTOSWAP': '0', 'VF_INOTIFYWAIT_BIN': inotifywait, 'VF_FFMPEG_BIN': ffmpeg, 'VF_FFPROBE_BIN': str(self.temp_dir / 'bin' / 'missing-ffprobe'), 'VF_PLAYOUT_DIR': playdir, 'READY_STABLE_POLLS': '0', 'READY_POLL_INTERVAL_SEC': '0.01', 'PRELOAD_EXISTING_ON_START': '0'})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('Switched to slot B', proc.stdout)
        self.assertEqual((playdir / '.active_slot').read_text(encoding='utf-8').strip(), 'B')
        self.assertEqual((playdir / 'current_B.mp4').read_bytes(), b'new clip data')

    def test_smoke_check_validates_tools_and_prints_urls(self):
        ffmpeg = self._make_fake_tool('ffmpeg')
        ffprobe = self._make_fake_tool('ffprobe')
        inotifywait = self._make_fake_tool('inotifywait')
        self._write_yaml('worqspace/config.yaml', {'run': {'output_dir': './custom-run'}, 'finalizer': {'per_cycle_interpolation': False}})
        proc = self._run_script('--smoke-check', extra_env={'OBS_AUTOSWAP': '0', 'VF_FFMPEG_BIN': ffmpeg, 'VF_FFPROBE_BIN': ffprobe, 'VF_INOTIFYWAIT_BIN': inotifywait})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('Smoke-check OK', proc.stdout)
        self.assertIn('SRT_URL_A=', proc.stdout)
        self.assertIn('SRT_URL_B=', proc.stdout)

    def test_validate_file_returns_nonzero_for_invalid_media_when_ffprobe_rejects(self):
        bad_file = self.temp_dir / 'bad.mp4'
        bad_file.write_bytes(b'not-a-real-video')
        ffprobe = self._make_fake_tool('ffprobe', body='exit 1\n')
        proc = self._run_script('--validate-file', str(bad_file), extra_env={'VF_FFPROBE_BIN': ffprobe, 'READY_STABLE_POLLS': '0', 'READY_POLL_INTERVAL_SEC': '0.01', 'READY_TIMEOUT_SEC': '1'})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('not ready/valid', proc.stderr)

    def test_reseed_slots_uses_newest_existing_clip(self):
        watch_dir = self.temp_dir / 'run' / 'videos'
        playdir = self.temp_dir / 'run' / 'obs'
        watch_dir.mkdir(parents=True, exist_ok=True)
        playdir.mkdir(parents=True, exist_ok=True)
        old_file = watch_dir / 'older.mp4'
        new_file = watch_dir / 'newer.mp4'
        old_file.write_bytes(b'old')
        new_file.write_bytes(b'newer-bytes')
        os.utime(old_file, (1_700_000_000, 1_700_000_000))
        os.utime(new_file, (1_800_000_000, 1_800_000_000))
        ffmpeg = self._make_fake_tool('ffmpeg')
        ffprobe = self._make_fake_tool('ffprobe', body='echo video\n')
        inotifywait = self._make_fake_tool('inotifywait')
        proc = self._run_script('--reseed-slots', extra_env={'OBS_AUTOSWAP': '0', 'VF_WATCH_DIR': watch_dir, 'VF_PLAYOUT_DIR': playdir, 'VF_FFMPEG_BIN': ffmpeg, 'VF_FFPROBE_BIN': ffprobe, 'VF_INOTIFYWAIT_BIN': inotifywait, 'READY_STABLE_POLLS': '0', 'READY_POLL_INTERVAL_SEC': '0.01'})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('Re-seeded both slots', proc.stdout)
        self.assertEqual((playdir / 'current_A.mp4').read_bytes(), b'newer-bytes')
        self.assertEqual((playdir / 'current_B.mp4').read_bytes(), b'newer-bytes')

    def test_script_contains_watch_and_health_monitor_hardening(self):
        content = (self.repo_root / 'vf-obs-watcher-srt-endpoints.sh').read_text(encoding='utf-8')
        self.assertIn('-e close_write,moved_to', content)
        self.assertIn('ffmpeg slot $slot is not running; restarting', content)
        self.assertIn('Watch dir missing, recreating', content)
        self.assertIn('--validate-file', content)
        self.assertIn('--reseed-slots', content)


if __name__ == '__main__':
    unittest.main()
