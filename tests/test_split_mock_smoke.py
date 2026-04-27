import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from vfaq.visual_faqtory import VisualFaQtory


class SplitMockSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix='vfaq_smoke_'))
        self.worqspace_dir = self.temp_dir / 'worqspace'
        self.worqspace_dir.mkdir(parents=True, exist_ok=True)
        (self.worqspace_dir / 'story.txt').write_text('Para one.\n\nPara two.', encoding='utf-8')
        config = {
            'backend': {'type': 'hybrid', 'width': 320, 'height': 180, 'mock_delay': 0.0},
            'image_backend': {'type': 'mock'},
            'video_backend': {'type': 'mock'},
            'morph_backend': {'type': 'mock'},
            'input': {'mode': 'text'},
            'paragraph_story': {
                'max_paragraphs': 2,
                'img2vid_duration_sec': 0.5,
                'video_fps': 2,
                'timing_authority': 'duration',
            },
            'finalizer': {'enabled': False, 'per_cycle_interpolation': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_run_completes_with_split_mock_backends(self):
        run_dir = self.temp_dir / 'run'
        saved_run = self.worqspace_dir / 'saved-runs' / 'split-mock-smoke'
        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=run_dir,
            dry_run=False,
            project_name='split-mock-smoke',
        )
        vf.run()
        self.assertTrue(saved_run.exists())
        self.assertTrue((saved_run / 'final_output.mp4').exists())
        self.assertTrue((saved_run / 'faqtory_state.json').exists())
        self.assertTrue((saved_run / 'videos' / 'video_001.mp4').exists())
        self.assertTrue((saved_run / 'videos' / 'video_002.mp4').exists())
        state = (saved_run / 'faqtory_state.json').read_text(encoding='utf-8')
        self.assertIn('split(image=mock, video=mock, morph=mock)', state)


if __name__ == '__main__':
    unittest.main()
