import tempfile
import unittest
from pathlib import Path

from vfaq.backends import (
    DelegatingBackend,
    GenerationRequest,
    GenerationResult,
    MockBackend,
    QwenImageComfyUIBackend,
    create_backend,
    describe_backend_config,
    extract_backend_config,
    get_backend_type_for_capability,
    resolve_capability_backend_configs,
)
from vfaq.venice_backend import VeniceBackend


class DummyVideoBackend(MockBackend):
    def __init__(self):
        super().__init__({'mock_delay': 0.0})
        self.name = 'dummy_video'

    def generate_extension(self, request, source_video):
        return GenerationResult(success=True, video_path=Path(source_video), metadata={'extended': True})


class BackendRoutingTests(unittest.TestCase):
    def test_extract_backend_config_supports_top_level_split_sections(self):
        root = {
            'backend': {'type': 'hybrid', 'width': 1024},
            'image_backend': {'type': 'qwen_image_comfyui', 'workflow_image': './qwen_t2i.json'},
            'video_backend': {'type': 'comfyui', 'workflow_video': './svd.json'},
            'morph_backend': {'type': 'comfyui', 'workflow_morph': './morph.json'},
        }
        backend_cfg = extract_backend_config(root)
        resolved = resolve_capability_backend_configs(root)
        self.assertEqual(backend_cfg['type'], 'hybrid')
        self.assertEqual(resolved['image']['type'], 'qwen_image_comfyui')
        self.assertEqual(resolved['video']['type'], 'comfyui')
        self.assertEqual(resolved['morph']['type'], 'comfyui')
        self.assertEqual(get_backend_type_for_capability(root, 'image'), 'qwen_image_comfyui')
        self.assertEqual(describe_backend_config(root), 'split(image=qwen_image_comfyui, video=comfyui, morph=comfyui)')

    def test_create_backend_keeps_single_backend_compatibility(self):
        backend = create_backend({'type': 'mock', 'mock_delay': 0.0})
        self.assertIsInstance(backend, MockBackend)
        self.assertEqual(backend.name, 'mock')

    def test_delegating_backend_routes_generate_extension_to_video_backend(self):
        router = DelegatingBackend(
            config={'type': 'hybrid'},
            image_backend=MockBackend({'mock_delay': 0.0}),
            video_backend=DummyVideoBackend(),
            morph_backend=MockBackend({'mock_delay': 0.0}),
            capability_configs={
                'image': {'type': 'mock'},
                'video': {'type': 'dummy_video'},
                'morph': {'type': 'mock'},
            },
        )
        tmp = Path(tempfile.mkdtemp(prefix='delegating_backend_'))
        src = tmp / 'source.mp4'
        src.write_bytes(b'1234')
        result = router.generate_extension(GenerationRequest(prompt='x', output_dir=tmp), src)
        self.assertTrue(result.success)
        self.assertEqual(result.video_path, src)
        self.assertTrue(result.metadata['extended'])

    def test_qwen_backend_requires_workflow_image_for_availability(self):
        backend = QwenImageComfyUIBackend({'api_url': 'http://localhost:8188'})
        available, message = backend.check_availability()
        self.assertFalse(available)
        self.assertIn('workflow_image', message)

    def test_extract_backend_config_keeps_venice_section(self):
        root = {
            'backend': {'type': 'hybrid'},
            'image_backend': {'type': 'mock'},
            'video_backend': {'type': 'venice'},
            'venice': {'video_model_text_to_video': 'wan-2.5-preview-text-to-video'},
        }
        backend_cfg = extract_backend_config(root)
        self.assertEqual(backend_cfg['venice']['video_model_text_to_video'], 'wan-2.5-preview-text-to-video')
        self.assertEqual(get_backend_type_for_capability(root, 'video'), 'venice')



    def test_create_backend_supports_hybrid_venice_routing(self):
        root = {
            'backend': {'type': 'hybrid'},
            'image_backend': {'type': 'mock'},
            'video_backend': {'type': 'venice'},
            'morph_backend': {'type': 'venice'},
            'venice': {'validate_models': False},
        }
        backend = create_backend(root)
        self.assertIsInstance(backend, DelegatingBackend)
        self.assertIsInstance(backend.video_backend, VeniceBackend)
        self.assertIsInstance(backend.morph_backend, VeniceBackend)
        self.assertIsInstance(backend.image_backend, MockBackend)

if __name__ == '__main__':
    unittest.main()
