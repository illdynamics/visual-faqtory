import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from vfaq.backends import DelegatingBackend, GenerationResult, MockBackend
from vfaq.run_state import RunState
from vfaq.visual_faqtory import VisualFaQtory


_MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
    0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
    0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
    0x44, 0xAE, 0x42, 0x60, 0x82,
])


class CountingMorphBackend(MockBackend):
    def __init__(self):
        super().__init__({'mock_delay': 0.0})
        self.morph_calls = []

    def generate_morph_video(self, request, start_image_path, end_image_path):
        self.morph_calls.append((Path(start_image_path), Path(end_image_path), request.atom_id))
        return super().generate_morph_video(request, start_image_path, end_image_path)


class ResumeAndLoopClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix='vfaq_resume_validation_'))
        self.worqspace_dir = self.temp_dir / 'worqspace'
        self.worqspace_dir.mkdir(parents=True, exist_ok=True)
        (self.worqspace_dir / 'story.txt').write_text('Para one.\n\nPara two.', encoding='utf-8')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _base_split_config(self):
        return {
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
                'enable_loop_closure': False,
            },
            'finalizer': {'enabled': False, 'per_cycle_interpolation': False},
        }

    def _seed_meta(self, run_dir, config, story_text='Para one.\n\nPara two.'):
        meta = run_dir / 'meta'
        meta.mkdir(parents=True, exist_ok=True)
        (meta / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        (meta / 'story.txt').write_text(story_text, encoding='utf-8')

    def _seed_cycle_artifacts(self, run_dir, cycle_indices):
        videos = run_dir / 'videos'
        frames = run_dir / 'frames'
        videos.mkdir(parents=True, exist_ok=True)
        frames.mkdir(parents=True, exist_ok=True)
        for idx in cycle_indices:
            (videos / f'video_{idx:03d}.mp4').write_bytes(f'video-{idx}'.encode('utf-8'))
            (frames / f'lastframe_{idx:03d}.png').write_bytes(_MINIMAL_PNG)
        (frames / 'anchor_frame_001.png').write_bytes(_MINIMAL_PNG)

    def test_finalization_only_resume_runs_without_regenerating_cycles_and_preserves_state(self):
        config = self._base_split_config()
        config['paragraph_story']['enable_loop_closure'] = False
        config['paragraph_story']['max_paragraphs'] = 1
        run_dir = self.temp_dir / 'run_finalization_resume'
        self._seed_meta(run_dir, config)
        self._seed_cycle_artifacts(run_dir, [1, 2])

        state = RunState(
            run_id='resume-finalize',
            status='running',
            backend_type='split(image=mock, video=mock, morph=mock)',
            cycles_planned=2,
            cycles_completed=2,
            next_cycle_index=3,
            last_completed_cycle=2,
            final_video_paths=[str(run_dir / 'videos' / 'video_001.mp4'), str(run_dir / 'videos' / 'video_002.mp4')],
            completed_cycle_indices=[1, 2],
            resume_enabled=True,
        )
        (run_dir / 'faqtory_state.json').write_text(json.dumps(state.to_dict(), indent=2), encoding='utf-8')

        finalized_calls = {'count': 0}

        def fake_finalize(self, cycle_video_paths):
            finalized_calls['count'] += 1
            self.final_output_path.write_bytes(b'final-output')
            return self.final_output_path

        def fake_post_finalize(self):
            self.final_deliverable_path.write_bytes(b'final-1080p')
            return self.final_deliverable_path

        with patch('vfaq.visual_faqtory.Finalizer.finalize', new=fake_finalize), \
             patch('vfaq.visual_faqtory.Finalizer.run_post_stitch_finalizer', new=fake_post_finalize):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=run_dir,
                dry_run=False,
                resume=True,
                project_name='resume-finalizer',
            )
            vf.run()

        self.assertEqual(finalized_calls['count'], 1)
        saved_run = self.worqspace_dir / 'saved-runs' / 'resume-finalizer'
        self.assertTrue((saved_run / 'final_output.mp4').exists())
        self.assertTrue((saved_run / 'final_60fps_1080p.mp4').exists())

        saved_state = json.loads((saved_run / 'faqtory_state.json').read_text(encoding='utf-8'))
        self.assertEqual(saved_state['status'], 'completed')
        self.assertEqual(saved_state['cycles_completed'], 2)
        self.assertEqual(saved_state['completed_cycle_indices'], [1, 2])
        self.assertEqual(len(saved_state['final_video_paths']), 2)
        self.assertEqual(saved_state['backend_type'], 'split(image=mock, video=mock, morph=mock)')
        self.assertEqual(Path(saved_state['saved_to']).resolve(), saved_run.resolve())

    def test_partial_progress_resume_starts_from_next_cycle_and_keeps_split_routing(self):
        current_worqspace_config = self._base_split_config()
        current_worqspace_config['backend'] = {'type': 'mock'}
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(current_worqspace_config, sort_keys=False), encoding='utf-8')

        split_resume_config = self._base_split_config()
        split_resume_config['image_backend'] = {'type': 'mock'}
        split_resume_config['video_backend'] = {'type': 'mock'}
        split_resume_config['morph_backend'] = {'type': 'mock'}
        run_dir = self.temp_dir / 'run_partial_resume'
        self._seed_meta(run_dir, split_resume_config)
        self._seed_cycle_artifacts(run_dir, [1])

        state = RunState(
            run_id='resume-partial',
            status='running',
            backend_type='split(image=mock, video=mock, morph=mock)',
            cycles_planned=2,
            cycles_completed=1,
            next_cycle_index=2,
            last_completed_cycle=1,
            final_video_paths=[str(run_dir / 'videos' / 'video_001.mp4')],
            completed_cycle_indices=[1],
            resume_enabled=True,
        )
        (run_dir / 'faqtory_state.json').write_text(json.dumps(state.to_dict(), indent=2), encoding='utf-8')

        calls = {}
        cycle1_path = run_dir / 'videos' / 'video_001.mp4'
        cycle1_before = cycle1_path.read_bytes()

        def fake_run_sliding_story(*, story_path, qodeyard_dir, config, max_cycles, base_image_path, base_video_path, **engine_kwargs):
            calls['start_cycle'] = engine_kwargs.get('start_cycle')
            calls['initial_final_video_paths'] = [str(p) for p in engine_kwargs.get('initial_final_video_paths', [])]
            calls['initial_completed_cycles'] = sorted(engine_kwargs.get('initial_completed_cycles', set()))
            calls['backend_summary'] = config.backend_config
            self.assertEqual(config.backend_config['type'], 'hybrid')
            self.assertEqual(config.backend_config['image_backend']['type'], 'mock')
            self.assertEqual(config.backend_config['video_backend']['type'], 'mock')
            self.assertEqual(config.backend_config['morph_backend']['type'], 'mock')
            self.assertEqual(engine_kwargs.get('start_cycle'), 2)
            checkpoint = engine_kwargs['checkpoint_callback']
            cycle2_video = qodeyard_dir / 'videos' / 'video_002.mp4'
            cycle2_lastframe = qodeyard_dir / 'frames' / 'lastframe_002.png'
            cycle2_video.parent.mkdir(parents=True, exist_ok=True)
            cycle2_lastframe.parent.mkdir(parents=True, exist_ok=True)
            cycle2_video.write_bytes(b'video-2')
            cycle2_lastframe.write_bytes(_MINIMAL_PNG)
            (qodeyard_dir / 'final_output.mp4').write_bytes(b'final-output')
            checkpoint(2, cycle2_lastframe, cycle2_video, qodeyard_dir / 'frames' / 'anchor_frame_001.png')
            return qodeyard_dir / 'final_output.mp4'

        with patch('vfaq.visual_faqtory.run_sliding_story', new=fake_run_sliding_story):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=run_dir,
                dry_run=False,
                resume=True,
                project_name='resume-partial',
            )
            vf.run()

        saved_run = self.worqspace_dir / 'saved-runs' / 'resume-partial'
        self.assertEqual(calls['start_cycle'], 2)
        self.assertEqual(calls['initial_completed_cycles'], [1])
        self.assertEqual(len(calls['initial_final_video_paths']), 1)
        self.assertEqual((saved_run / 'videos' / 'video_001.mp4').read_bytes(), cycle1_before)
        self.assertTrue((saved_run / 'videos' / 'video_002.mp4').exists())

        saved_state = json.loads((saved_run / 'faqtory_state.json').read_text(encoding='utf-8'))
        self.assertEqual(saved_state['cycles_completed'], 2)
        self.assertEqual(saved_state['completed_cycle_indices'], [1, 2])
        self.assertEqual(saved_state['backend_type'], 'split(image=mock, video=mock, morph=mock)')

    def test_loop_closure_with_split_backends_routes_to_morph_backend(self):
        config = self._base_split_config()
        config['paragraph_story']['enable_loop_closure'] = True
        (self.worqspace_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

        morph_backend = CountingMorphBackend()
        router = DelegatingBackend(
            config={'type': 'hybrid'},
            image_backend=MockBackend({'mock_delay': 0.0}),
            video_backend=MockBackend({'mock_delay': 0.0}),
            morph_backend=morph_backend,
            capability_configs={
                'image': {'type': 'mock'},
                'video': {'type': 'mock'},
                'morph': {'type': 'mock'},
            },
        )

        with patch('vfaq.sliding_story_engine.create_backend', return_value=router):
            vf = VisualFaQtory(
                worqspace_dir=self.worqspace_dir,
                run_dir=self.temp_dir / 'run_loop_closure',
                dry_run=False,
                project_name='loop-closure-split',
            )
            vf.run()

        saved_run = self.worqspace_dir / 'saved-runs' / 'loop-closure-split'
        self.assertEqual(len(morph_backend.morph_calls), 1)
        self.assertTrue(any((saved_run / 'videos').glob('video_loop_*.mp4')))
        saved_state = json.loads((saved_run / 'faqtory_state.json').read_text(encoding='utf-8'))
        self.assertEqual(saved_state['backend_type'], 'split(image=mock, video=mock, morph=mock)')


if __name__ == '__main__':
    unittest.main()
