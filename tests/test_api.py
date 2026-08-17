import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    # isolate the sqlite db used by tests from the dev db
    tmp_db = os.path.join(tempfile.gettempdir(), "amr_test.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    monkeypatch.setenv("AMR_TEST_DB", tmp_db)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_route_endpoint_returns_valid_response(client):
    resp = client.post("/route", json={"prompt": "Convert 25 USD to EUR."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["difficulty"] >= 0.0
    assert "explanation" in body


def test_route_endpoint_hard_task_escalates(client):
    resp = client.post(
        "/route",
        json={
            "prompt": "Find the subtle concurrency bugs in this distributed Python system. "
            "Explain the race conditions and propose a safe redesign:\n"
            "```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n"
            "    def increment(self):\n        current = self.value\n        time.sleep(0.001)\n"
            "        self.value = current + 1\n```\n"
            "This is called concurrently from 50 worker threads and the final count is wrong."
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] in ("claude-opus-5", "claude-fable-5")


def test_route_endpoint_respects_constraints(client):
    resp = client.post(
        "/route",
        json={"prompt": "Convert 25 USD to EUR.", "constraints": {"max_cost": 0.5}},
    )
    assert resp.status_code == 200


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 4


def test_benchmark_endpoint(client):
    resp = client.get("/evaluate/benchmark")
    assert resp.status_code == 200
    body = resp.json()
    assert "routing_accuracy" in body
    assert "underpowered_rate" in body
    assert "overkill_rate" in body


def test_tournament_endpoint(client):
    resp = client.post("/evaluate/tournament", json={"prompt": "Summarize this paragraph."})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 4


def test_feedback_endpoint(client):
    route_resp = client.post("/route", json={"prompt": "Summarize this paragraph."})
    record_id = route_resp.json()["record_id"]
    fb_resp = client.post(
        "/feedback",
        json={"record_id": record_id, "success": True, "actual_result_quality": 0.9},
    )
    assert fb_resp.status_code == 200
