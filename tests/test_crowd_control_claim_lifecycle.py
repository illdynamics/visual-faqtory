import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vfaq.crowd_control.client import CrowdClient
from vfaq.crowd_control.db import CrowdDB
from vfaq.crowd_control.models import CrowdControlConfig, SubmissionStatus
from vfaq.crowd_control.server import create_crowd_app


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class CrowdControlClaimLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="vfaq_crowd_claims_"))
        self.db_path = self.temp_dir / "crowd.sqlite3"

    def tearDown(self):
        for path in sorted(self.temp_dir.glob("**/*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    def test_config_invalid_bake_mode_falls_back_to_default(self):
        cfg = CrowdControlConfig.from_dict({"enabled": True, "bake_mode": "not_a_mode"})
        self.assertEqual(cfg.bake_mode, "reinject_keyframe")

    def test_claim_ack_requeue_and_stale_reclaim(self):
        db = CrowdDB(self.db_path)
        first_id = db.enqueue("127.0.0.1", "first prompt")
        second_id = db.enqueue("127.0.0.1", "second prompt")

        claim1 = db.claim_next(claim_timeout_seconds=900)
        self.assertIsNotNone(claim1)
        self.assertEqual(claim1["id"], first_id)
        self.assertTrue(db.ack_served(first_id, claim_id=claim1["claim_id"]))

        claim2 = db.claim_next(claim_timeout_seconds=900)
        self.assertEqual(claim2["id"], second_id)
        self.assertTrue(db.requeue_claimed(second_id, reason="forced retry", claim_id=claim2["claim_id"]))

        reclaimed = db.claim_next(claim_timeout_seconds=900)
        self.assertEqual(reclaimed["id"], second_id)

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=1800)).isoformat()
        conn = db._connect()
        try:
            conn.execute(
                "UPDATE submissions SET status = ?, claimed_at = ?, claim_id = ? WHERE id = ?",
                (SubmissionStatus.CLAIMED.value, stale_time, "stale-claim", second_id),
            )
            conn.commit()
        finally:
            conn.close()

        stale_reclaim = db.claim_next(claim_timeout_seconds=10)
        self.assertEqual(stale_reclaim["id"], second_id)

    def test_server_next_returns_id_and_prompt_and_supports_ack(self):
        cfg = CrowdControlConfig(
            enabled=True,
            base_url="http://127.0.0.1:8808/visuals",
            pop_token="secret-token",
            db_path=str(self.db_path),
            prefix="/visuals",
        )
        app = create_crowd_app(cfg)
        client = TestClient(app)

        submit = client.post("/visuals/api/submit", json={"prompt": "mutate with amber haze"})
        self.assertEqual(submit.status_code, 200)
        self.assertTrue(submit.json().get("ok"))

        next_resp = client.get("/visuals/api/next", headers={"Authorization": "Bearer secret-token"})
        self.assertEqual(next_resp.status_code, 200)
        body = next_resp.json()
        self.assertEqual(body["prompt"], "mutate with amber haze")
        self.assertIsInstance(body.get("id"), int)
        self.assertTrue(body.get("claim_id"))

        ack_resp = client.post(
            "/visuals/api/ack",
            headers={"Authorization": "Bearer secret-token"},
            json={"id": body["id"], "claim_id": body["claim_id"]},
        )
        self.assertEqual(ack_resp.status_code, 200)
        self.assertTrue(ack_resp.json().get("ok"))

    def test_client_pop_next_backwards_compatible(self):
        cfg = CrowdControlConfig(enabled=True, base_url="http://localhost:8808/visuals", pop_token="tok")
        cc = CrowdClient(cfg)

        with patch("vfaq.crowd_control.client.requests.get") as get_mock, patch("vfaq.crowd_control.client.requests.post") as post_mock:
            get_mock.return_value = _FakeResponse(
                status_code=200,
                payload={"id": 42, "prompt": "crowd text", "claim_id": "claim-42"},
            )
            post_mock.return_value = _FakeResponse(status_code=200, payload={"ok": True, "id": 42})

            prompt = cc.pop_next()

        self.assertEqual(prompt, "crowd text")
        self.assertEqual(get_mock.call_count, 1)
        self.assertEqual(post_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
