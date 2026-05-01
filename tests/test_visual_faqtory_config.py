import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from vfaq.visual_faqtory import VisualFaQtory


class VisualFaQtoryConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix='vfaq_cfg_'))
        self.worqspace_dir = self.temp_dir / 'worqspace'
        self.worqspace_dir.mkdir(parents=True, exist_ok=True)
        (self.worqspace_dir / 'story.txt').write_text('Paragraph one.\n\nParagraph two.', encoding='utf-8')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_story_config_supports_split_root_sections(self):
        config = {
            'backend': {'type': 'hybrid', 'width': 768, 'height': 432},
            'image_backend': {'type': 'qwen_image_comfyui', 'workflow_image': './worqspace/workflows/qwen_t2i.json'},
            'video_backend': {'type': 'mock'},
            'morph_backend': {'type': 'comfyui', 'workflow_morph': './worqspace/workflows/morph.json'},
            'paragraph_story': {'enable_loop_closure': True, 'img2vid_duration_sec': 2.0, 'video_fps': 8, 'timing_authority': 'duration'},
            'input': {'mode': 'text'},
            'finalizer': {'enabled': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=self.temp_dir / 'run',
            dry_run=True,
            project_name='test-project',
        )
        story_config = vf._build_story_config()

        self.assertEqual(story_config.backend_config['type'], 'hybrid')
        self.assertEqual(story_config.backend_config['image_backend']['type'], 'qwen_image_comfyui')
        self.assertEqual(story_config.backend_config['video_backend']['type'], 'mock')
        self.assertEqual(story_config.backend_config['morph_backend']['type'], 'comfyui')
        self.assertTrue(story_config.enable_loop_closure)

    def test_build_story_config_uses_venice_duration_defaults(self):
        config = {
            'backend': {'type': 'venice', 'width': 1024, 'height': 576},
            'venice': {'video_duration': '5s'},
            'paragraph_story': {},
            'input': {'mode': 'text'},
            'finalizer': {'enabled': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=self.temp_dir / 'run_venice',
            dry_run=True,
            project_name='venice-project',
        )
        story_config = vf._build_story_config()

        self.assertEqual(story_config.img2vid_duration_sec, 5.0)
        self.assertEqual(story_config.venice_config['video_duration'], '5s')

    def test_invalid_yaml_config_fails_fast_instead_of_falling_back_to_mock(self):
        (self.worqspace_dir / 'config.yaml').write_text(
            '''backend:
  type: hybrid
  morph_backend:
    type: animatediff
      workflow_morph: broken.yaml
''',
            encoding='utf-8',
        )

        with self.assertRaises(RuntimeError) as ctx:
            VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=self.temp_dir / 'run_invalid_yaml',
                dry_run=True,
                project_name='invalid-yaml-project',
            )

        self.assertIn('Failed to parse', str(ctx.exception))
        self.assertIn('silently route to mock backends', str(ctx.exception))

    def test_build_story_config_uses_nested_venice_duration_defaults(self):
        config = {
            'backend': {'type': 'venice', 'width': 1024, 'height': 576},
            'venice': {'video': {'duration_seconds': 10}},
            'paragraph_story': {},
            'input': {'mode': 'text'},
            'finalizer': {'enabled': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=self.temp_dir / 'run_venice_nested',
            dry_run=True,
            project_name='venice-project-nested',
        )
        story_config = vf._build_story_config()

        self.assertEqual(story_config.img2vid_duration_sec, 10.0)
        self.assertEqual(story_config.venice_config['video']['duration_seconds'], 10)

    def test_build_story_config_respects_yaml_reinject_when_cli_not_set(self):
        config = {
            'backend': {'type': 'mock'},
            'paragraph_story': {'reinject': False},
            'input': {'mode': 'text'},
            'finalizer': {'enabled': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=self.temp_dir / 'run_reinject_yaml',
            dry_run=True,
            project_name='reinject-yaml',
            reinject=None,
        )
        story_config = vf._build_story_config()
        self.assertFalse(story_config.reinject)

    def test_build_story_config_cli_reinject_override_wins(self):
        config = {
            'backend': {'type': 'mock'},
            'paragraph_story': {'reinject': True},
            'input': {'mode': 'text'},
            'finalizer': {'enabled': False},
        }
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        vf = VisualFaQtory(
            worqspace_dir=self.worqspace_dir,
            run_dir=self.temp_dir / 'run_reinject_cli',
            dry_run=True,
            project_name='reinject-cli',
            reinject=False,
        )
        story_config = vf._build_story_config()
        self.assertFalse(story_config.reinject)


if __name__ == '__main__':
    unittest.main()
