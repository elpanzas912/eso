import sys
import time
from unittest.mock import MagicMock, patch
import json

# Prevent ImportError for heavy/optional deps when app.py is imported
for mod in ["whisper", "torch", "yt_dlp", "rapidfuzz"]:
    sys.modules[mod] = MagicMock()

from app import app, _pending, _pending_lock, _HANDOFF_TTL  # noqa: E402


def test_process_endpoint_accepts_episode_id_and_series_name():
    client = app.test_client()
    with patch("app._active_job_count", return_value=0), patch("app._run_job"):
        resp = client.post(
            "/api/process",
            json={
                "short_url": "https://youtube.com/shorts/abc",
                "long_url": "https://youtube.com/watch?v=def",
                "episode_id": "S01E01",
                "series_name": "Test Show",
                "audio_index": "2",
                "model": "base",
            },
        )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "job_id" in data


def test_process_endpoint_rejects_invalid_audio_index():
    client = app.test_client()
    resp = client.post(
        "/api/process",
        json={
            "short_url": "https://youtube.com/shorts/abc",
            "long_url": "https://youtube.com/watch?v=def",
            "audio_index": "not-a-number",
        },
    )
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "audio_index" in data.get("error", "").lower()


def test_export_endpoint_rejects_invalid_audio_index():
    client = app.test_client()
    resp = client.post(
        "/api/export",
        json={"job_id": "12345678", "segments": [], "audio_index": "bad"},
    )
    assert resp.status_code == 400


def test_handoff_ttl():
    with _pending_lock:
        _pending.clear()
    key = "testkey123"
    _pending[key] = {
        "data": {"foo": "bar"},
        "created_at": time.time() - _HANDOFF_TTL - 1,
    }

    with app.test_client() as client:
        resp = client.get(f"/api/handoff/{key}")
        assert resp.status_code == 404


def test_handoff_save_and_load():
    with _pending_lock:
        _pending.clear()
    with app.test_client() as client:
        save_resp = client.post("/api/handoff", json={"video_path": "/tmp/v.mp4"})
        assert save_resp.status_code == 200
        save_data = json.loads(save_resp.data)
        key = save_data["key"]

        load_resp = client.get(f"/api/handoff/{key}")
        assert load_resp.status_code == 200
        load_data = json.loads(load_resp.data)
        assert load_data["video_path"] == "/tmp/v.mp4"

        # Second load should fail (pop semantics)
        load_resp2 = client.get(f"/api/handoff/{key}")
        assert load_resp2.status_code == 404
