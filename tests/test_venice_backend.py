import base64
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vfaq.backends import GenerationRequest, InputMode
from vfaq.venice_backend import VeniceBackend, VeniceConfig


_MINIMAL_PNG = base64.b64encode(bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
    0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
    0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
    0x44, 0xAE, 0x42, 0x60, 0x82
])).decode('utf-8')


class FakeResponse:
    def __init__(self, status_code=200, headers=None, content=b'', json_data=None):
        self.status_code = status_code
        self.headers = headers or {'Content-Type': 'application/json'}
        self.content = content
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError('no json')
        return self._json_data


class VeniceBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix='venice_backend_'))
        self.output_dir = self.temp_dir / 'out'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_image = self.temp_dir / 'source.png'
        self.source_image.write_bytes(base64.b64decode(_MINIMAL_PNG))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nested_schema_and_env_placeholder_are_supported(self):
        with patch.dict(os.environ, {'VENICE_API_KEY': 'env-key'}, clear=True):
            cfg = VeniceConfig.from_dict({
                'api_key': '${VENICE_API_KEY}',
                'models': {
                    'text2img': 'z-image-turbo',
                    'img2img': 'qwen-edit',
                    'text2vid': 'wan-2.5-preview-text-to-video',
                    'img2vid': 'wan-2.5-preview-image-to-video',
                },
                'image': {'width': 832, 'height': 480, 'cfg_scale': 6.5, 'steps': 12},
                'video': {'duration_seconds': 10, 'resolution': '1080p', 'aspect_ratio': '16:9'},
            })

        self.assertEqual(cfg.api_key, 'env-key')
        self.assertEqual(cfg.image_model, 'z-image-turbo')
        self.assertEqual(cfg.image_edit_model, 'qwen-edit')
        self.assertEqual(cfg.video_model_text_to_video, 'wan-2.5-preview-text-to-video')
        self.assertEqual(cfg.video_model_image_to_video, 'wan-2.5-preview-image-to-video')
        self.assertEqual(cfg.image_width, 832)
        self.assertEqual(cfg.image_height, 480)
        self.assertEqual(cfg.video_duration, '10s')
        self.assertEqual(cfg.video_resolution, '1080p')

    def test_check_availability_requires_api_key(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        with patch.dict(os.environ, {}, clear=True):
            ok, message = backend.check_availability()
        self.assertFalse(ok)
        self.assertIn('VENICE_API_KEY', message)

    def test_generate_image_text2img_writes_png(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        responses = [
            FakeResponse(json_data={'images': [_MINIMAL_PNG]}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_image(
                GenerationRequest(
                    prompt='A neon canal city',
                    mode=InputMode.TEXT,
                    width=1024,
                    height=576,
                    output_dir=self.output_dir,
                    atom_id='cycle_001',
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(result.image_path.exists())
        self.assertEqual(result.image_path.suffix, '.png')
        self.assertEqual(calls[0][2]['model'], 'z-image-turbo')
        self.assertEqual(result.metadata['response']['id'], None)

    def test_generate_image_uses_nested_default_dimensions(self):
        backend = VeniceBackend({'venice': {
            'validate_models': False,
            'image': {'width': 896, 'height': 512},
        }})
        responses = [FakeResponse(json_data={'images': [_MINIMAL_PNG]})]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_image(
                GenerationRequest(
                    prompt='A neon canal city',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='cycle_003',
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(calls[0][2]['width'], 896)
        self.assertEqual(calls[0][2]['height'], 512)

    def test_generate_image_img2img_uses_edit_endpoint(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        responses = [
            FakeResponse(headers={'Content-Type': 'image/png'}, content=base64.b64decode(_MINIMAL_PNG)),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_image(
                GenerationRequest(
                    prompt='Add bioluminescent fog',
                    mode=InputMode.IMAGE,
                    init_image_path=self.source_image,
                    width=1024,
                    height=576,
                    output_dir=self.output_dir,
                    atom_id='cycle_002',
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(result.image_path.exists())
        self.assertIn('/image/edit', calls[0][1])
        self.assertEqual(calls[0][2]['model'], 'qwen-edit')

    def test_request_retries_rate_limit_and_honors_reset_header(self):
        backend = VeniceBackend({'venice': {'validate_models': False, 'max_retries': 2}})
        reset_at = str(time.time() + 0.2)
        responses = [
            FakeResponse(status_code=429, headers={'Content-Type': 'application/json', 'x-ratelimit-reset-requests': reset_at}, json_data={'code': 'RATE_LIMIT_EXCEEDED', 'message': 'slow down'}),
            FakeResponse(json_data={'images': [_MINIMAL_PNG]}),
        ]
        calls = []
        sleeps = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request), \
             patch('vfaq.venice_backend.time.sleep', side_effect=lambda s: sleeps.append(s)):
            result = backend.generate_image(
                GenerationRequest(
                    prompt='Retry me',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='cycle_retry',
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(len(calls), 2)
        self.assertTrue(sleeps and sleeps[0] >= 0)

    def test_generate_video_text2video_snaps_duration_to_supported_option_and_cleans_up(self):
        backend = VeniceBackend({'venice': {'validate_models': False, 'cleanup_after_download': True}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'0' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'issues': [
                        {
                            'code': 'invalid_enum_value',
                            'message': 'Duration not supported',
                            'path': ['duration'],
                            'options': ['5s'],
                        }
                    ]
                },
            ),
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q123'}),
            FakeResponse(json_data={'status': 'PROCESSING'}, headers={'Content-Type': 'application/json'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request), \
             patch('vfaq.venice_backend.time.sleep', return_value=None):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='A spectral rave in the canals',
                    mode=InputMode.TEXT,
                    duration_seconds=4.0,
                    width=1024,
                    height=576,
                    output_dir=self.output_dir,
                    atom_id='video_001',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        self.assertTrue(result.video_path.exists())
        self.assertEqual(calls[0][2]['duration'], '4s')
        self.assertEqual(calls[1][2]['duration'], '5s')
        self.assertIn('/video/complete', calls[-1][1])
        self.assertEqual(result.metadata['response']['retrieve_polls'], 2)
        self.assertEqual(result.metadata['operation'], 'text2vid')

    def test_generate_video_image_to_video_uses_reference_image(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'1' * 32
        responses = [
            FakeResponse(json_data={'model': backend.venice_cfg.video_model_image_to_video, 'queue_id': 'q456'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Calm camera drift over floating lanterns',
                    mode=InputMode.IMAGE,
                    width=1024,
                    height=576,
                    output_dir=self.output_dir,
                    atom_id='video_002',
                ),
                source_image=self.source_image,
            )

        self.assertTrue(result.success)
        payload = calls[0][2]
        self.assertEqual(payload['model'], backend.venice_cfg.video_model_image_to_video)
        self.assertTrue(payload['image_url'].startswith('data:image/png;base64,'))
        self.assertEqual(result.metadata['operation'], 'img2vid')



    def test_generate_video_payload_includes_resolution_when_supported(self):
        backend = VeniceBackend({'venice': {'validate_models': True}})
        backend._model_cache = [
            {
                'id': 'wan-2.5-preview-text-to-video',
                'type': 'video',
                'supportedResolutions': ['720p', '1080p'],
            }
        ]
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'2' * 32
        responses = [
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q789'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Keep supported optionals',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_003a',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        queue_payload = calls[0][2]
        self.assertIn('resolution', queue_payload)
        self.assertEqual(queue_payload['resolution'], '720p')

    def test_generate_video_omits_model_unsupported_optional_fields(self):
        backend = VeniceBackend({'venice': {'validate_models': True, 'video_audio': True}})
        backend._model_cache = [
            {
                'id': 'wan-2.5-preview-text-to-video',
                'type': 'video',
                'supportsAudioConfig': False,
                'supportedAspectRatios': [],
                'supportedResolutions': [],
            }
        ]
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'2' * 32
        responses = [
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q789'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='No unsupported optionals please',
                    mode=InputMode.TEXT,
                    width=1024,
                    height=576,
                    output_dir=self.output_dir,
                    atom_id='video_003',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        queue_payload = calls[0][2]
        self.assertNotIn('audio', queue_payload)
        self.assertNotIn('aspect_ratio', queue_payload)
        self.assertNotIn('resolution', queue_payload)
        self.assertEqual(
            result.metadata['response']['omitted_optional_fields'],
            ['aspect_ratio', 'resolution', 'audio'],
        )

    def test_generate_video_retries_queue_without_invalid_optional_field(self):
        backend = VeniceBackend({'venice': {'validate_models': False, 'video_audio': True}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'3' * 32
        responses = [
            FakeResponse(status_code=400, json_data={'error': 'invalid_request', 'message': 'Unsupported parameter: audio'}),
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q999'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Retry queue sans audio',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_retry',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        self.assertIn('audio', calls[0][2])
        self.assertNotIn('audio', calls[1][2])
        self.assertEqual(result.metadata['response']['stripped_retry_fields'], ['audio'])

    def test_generate_video_retries_queue_without_unsupported_aspect_ratio(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'4' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'issues': [
                        {
                            'code': 'custom',
                            'message': 'This model does not support aspect_ratio',
                            'path': ['aspect_ratio'],
                        }
                    ]
                },
            ),
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q_aspect'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Retry queue sans aspect ratio',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_aspect_retry',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        self.assertIn('aspect_ratio', calls[0][2])
        self.assertNotIn('aspect_ratio', calls[1][2])
        self.assertEqual(result.metadata['response']['stripped_retry_fields'], ['aspect_ratio'])

    def test_generate_video_learns_unsupported_aspect_ratio_per_model_and_op(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'4a' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'issues': [
                        {
                            'code': 'custom',
                            'message': 'This model does not support aspect_ratio',
                            'path': ['aspect_ratio'],
                        }
                    ]
                },
            ),
            FakeResponse(json_data={'model': backend.venice_cfg.video_model_image_to_video, 'queue_id': 'q_aspect_1'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
            FakeResponse(json_data={'model': backend.venice_cfg.video_model_image_to_video, 'queue_id': 'q_aspect_2'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result_1 = backend.generate_video(
                GenerationRequest(
                    prompt='first call learns unsupported aspect ratio',
                    mode=InputMode.IMAGE,
                    output_dir=self.output_dir,
                    atom_id='video_aspect_cache_1',
                ),
                source_image=self.source_image,
            )
            result_2 = backend.generate_video(
                GenerationRequest(
                    prompt='second call should omit aspect ratio immediately',
                    mode=InputMode.IMAGE,
                    output_dir=self.output_dir,
                    atom_id='video_aspect_cache_2',
                ),
                source_image=self.source_image,
            )

        self.assertTrue(result_1.success)
        self.assertTrue(result_2.success)
        queue_calls = [c for c in calls if '/video/queue' in c[1]]
        self.assertEqual(len(queue_calls), 3)
        self.assertIn('aspect_ratio', queue_calls[0][2])
        self.assertNotIn('aspect_ratio', queue_calls[1][2])
        self.assertNotIn('aspect_ratio', queue_calls[2][2])
        self.assertEqual(result_1.metadata['response']['payload_retry_count'], 1)
        self.assertEqual(result_2.metadata['response']['payload_retry_count'], 0)
        self.assertEqual(result_1.metadata['response']['stripped_retry_fields'], ['aspect_ratio'])
        self.assertIn('aspect_ratio', result_2.metadata['response']['omitted_optional_fields'])

    def test_generate_video_learns_unsupported_reference_images_per_model_and_op(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'4b' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'issues': [
                        {
                            'code': 'custom',
                            'message': 'This model does not support reference_image_urls',
                            'path': ['reference_image_urls'],
                        }
                    ]
                },
            ),
            FakeResponse(json_data={'model': backend.venice_cfg.video_model_text_to_video, 'queue_id': 'q_refs_1'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
            FakeResponse(json_data={'model': backend.venice_cfg.video_model_text_to_video, 'queue_id': 'q_refs_2'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result_1 = backend.generate_video(
                GenerationRequest(
                    prompt='first call learns unsupported reference image urls',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_refs_cache_1',
                    reference_image_paths=[self.source_image],
                ),
                source_image=None,
            )
            result_2 = backend.generate_video(
                GenerationRequest(
                    prompt='second call should omit reference image urls immediately',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_refs_cache_2',
                    reference_image_paths=[self.source_image],
                ),
                source_image=None,
            )

        self.assertTrue(result_1.success)
        self.assertTrue(result_2.success)
        queue_calls = [c for c in calls if '/video/queue' in c[1]]
        self.assertEqual(len(queue_calls), 3)
        self.assertIn('reference_image_urls', queue_calls[0][2])
        self.assertNotIn('reference_image_urls', queue_calls[1][2])
        self.assertNotIn('reference_image_urls', queue_calls[2][2])
        self.assertEqual(result_1.metadata['response']['payload_retry_count'], 1)
        self.assertEqual(result_2.metadata['response']['payload_retry_count'], 0)
        self.assertEqual(result_1.metadata['response']['stripped_retry_fields'], ['reference_image_urls'])
        self.assertIn('reference_image_urls', result_2.metadata['response']['omitted_optional_fields'])

    def test_generate_video_retries_queue_without_unsupported_resolution(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'5' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'issues': [
                        {
                            'code': 'custom',
                            'message': 'This model does not support resolution',
                            'path': ['resolution'],
                        }
                    ]
                },
            ),
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q_resolution'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Retry queue sans resolution',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_resolution_retry',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        first_payload = calls[0][2]
        retry_payload = calls[1][2]
        self.assertEqual(
            first_payload,
            {
                'model': 'wan-2.5-preview-text-to-video',
                'prompt': 'Retry queue sans resolution',
                'duration': '5s',
                'negative_prompt': 'low resolution, error, worst quality, low quality, defects',
                'aspect_ratio': '16:9',
                'resolution': '720p',
                'audio': False,
            },
        )
        self.assertEqual(
            retry_payload,
            {
                'model': 'wan-2.5-preview-text-to-video',
                'prompt': 'Retry queue sans resolution',
                'duration': '5s',
                'negative_prompt': 'low resolution, error, worst quality, low quality, defects',
                'aspect_ratio': '16:9',
                'audio': False,
            },
        )
        self.assertEqual(result.metadata['response']['stripped_retry_fields'], ['resolution'])

    def test_generate_video_retries_without_multiple_unsupported_optional_fields(self):
        backend = VeniceBackend({'venice': {'validate_models': False, 'video_audio': True}})
        video_bytes = b'\x00\x00\x00\x20ftypisom' + b'6' * 32
        responses = [
            FakeResponse(
                status_code=400,
                json_data={
                    'details': {
                        'resolution': {'_errors': ['This model does not support resolution']},
                    },
                    'issues': [
                        {
                            'code': 'custom',
                            'message': 'This model does not support audio',
                            'path': ['audio'],
                        },
                        {
                            'code': 'custom',
                            'message': 'This model does not support aspect_ratio',
                            'path': ['aspect_ratio'],
                        },
                    ],
                },
            ),
            FakeResponse(json_data={'model': 'wan-2.5-preview-text-to-video', 'queue_id': 'q_multi'}),
            FakeResponse(headers={'Content-Type': 'video/mp4'}, content=video_bytes),
            FakeResponse(json_data={'success': True}),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Retry queue sans multiple optional fields',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_multi_retry',
                ),
                source_image=None,
            )

        self.assertTrue(result.success)
        retry_payload = calls[1][2]
        self.assertNotIn('audio', retry_payload)
        self.assertNotIn('aspect_ratio', retry_payload)
        self.assertNotIn('resolution', retry_payload)
        self.assertEqual(
            result.metadata['response']['stripped_retry_fields'],
            ['audio', 'aspect_ratio', 'resolution'],
        )

    def test_generate_video_retry_failure_reports_removed_fields(self):
        backend = VeniceBackend({'venice': {'validate_models': False, 'video_audio': True}})
        responses = [
            FakeResponse(
                status_code=400,
                json_data={'error': 'invalid_request', 'message': 'Unsupported parameter: audio'},
            ),
            FakeResponse(
                status_code=400,
                json_data={'error': 'invalid_request', 'message': 'Queue request still invalid after retry'},
            ),
        ]
        calls = []

        def fake_request(method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, dict(json) if isinstance(json, dict) else json))
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True), \
             patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_video(
                GenerationRequest(
                    prompt='Fail after optional retry',
                    mode=InputMode.TEXT,
                    output_dir=self.output_dir,
                    atom_id='video_retry_fail',
                ),
                source_image=None,
            )

        self.assertFalse(result.success)
        self.assertEqual(len([c for c in calls if '/video/queue' in c[1]]), 2)
        self.assertIn('after retry without optional field(s): audio', result.error)
        self.assertIn('original queue error', result.error)

    def test_generate_morph_video_fails_cleanly_when_model_lacks_end_image_support(self):
        backend = VeniceBackend({'venice': {'validate_models': True}})
        backend._model_cache = [
            {
                'id': backend.venice_cfg.video_model_image_to_video,
                'type': 'video',
                'supportsEndImage': False,
            }
        ]

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True):
            result = backend.generate_morph_video(
                GenerationRequest(
                    prompt='Loop it back',
                    mode=InputMode.IMAGE,
                    output_dir=self.output_dir,
                    atom_id='video_morph',
                ),
                start_image_path=self.source_image,
                end_image_path=self.source_image,
            )

        self.assertFalse(result.success)
        self.assertIn('does not advertise end-frame morph support', result.error)

    def test_edit_image_json_error_is_reported_cleanly(self):
        backend = VeniceBackend({'venice': {'validate_models': False}})
        responses = [
            FakeResponse(status_code=400, headers={'Content-Type': 'application/json'}, json_data={'error': 'invalid_request', 'message': 'bad image'}),
        ]

        def fake_request(method, url, headers=None, json=None, timeout=None):
            return responses.pop(0)

        with patch.dict(os.environ, {'VENICE_API_KEY': 'test-key'}, clear=True),              patch.object(backend.session, 'request', side_effect=fake_request):
            result = backend.generate_image(
                GenerationRequest(
                    prompt='Break the edit path',
                    mode=InputMode.IMAGE,
                    init_image_path=self.source_image,
                    output_dir=self.output_dir,
                    atom_id='edit_error',
                )
            )

        self.assertFalse(result.success)
        self.assertIn('bad image', result.error)

if __name__ == '__main__':
    unittest.main()
