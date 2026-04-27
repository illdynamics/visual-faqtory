from pathlib import Path
from unittest.mock import patch

from vfaq.sliding_story_engine import _extract_last_frame_ffmpeg


def test_last_frame_extraction_uses_first_frame_for_single_frame_clip(tmp_path):
    video_path = tmp_path / 'single_frame.mp4'
    video_path.write_bytes(b'fake')
    output_path = tmp_path / 'last.png'

    with patch('vfaq.sliding_story_engine._probe_video_metadata', return_value={'duration': 0.25, 'fps': 4.0, 'nb_frames': 1}), \
         patch('vfaq.sliding_story_engine._extract_first_frame_ffmpeg') as first_frame:
        _extract_last_frame_ffmpeg(video_path, output_path)

    first_frame.assert_called_once_with(video_path, output_path)


def test_last_frame_extraction_falls_back_to_first_frame_when_all_strategies_fail(tmp_path):
    video_path = tmp_path / 'multi_frame.mp4'
    video_path.write_bytes(b'fake')
    output_path = tmp_path / 'last.png'

    with patch('vfaq.sliding_story_engine._probe_video_metadata', return_value={'duration': 2.0, 'fps': 4.0, 'nb_frames': 8}), \
         patch('vfaq.sliding_story_engine._run_ffmpeg_frame_extract', return_value=False), \
         patch('vfaq.sliding_story_engine._extract_first_frame_ffmpeg') as first_frame:
        _extract_last_frame_ffmpeg(video_path, output_path)

    first_frame.assert_called_once_with(video_path, output_path)
