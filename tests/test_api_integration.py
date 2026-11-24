"""Integration tests: call the running Machine Guard API over HTTP, as a real client would.

Start the API first (in a second terminal, from the repo root):
    export API_KEY=dev-key
    uvicorn app.main:app --port 8000
Then run:
    API_KEY=dev-key pytest -m integration

Settings come from environment variables:
    API_URL         default http://localhost:8000 (set it to your Render URL to test the live service)
    API_KEY         the key the API was started with (required)
    PREDICTION_LOG  default logs/predictions.db; the log test is skipped if this file isn't local
"""
import os
import sqlite3
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", "logs/predictions.db"))


@pytest.fixture(scope="module")
def api():
    try:
        httpx.get(f"{API_URL}/health", timeout=30)
    except httpx.HTTPError as exc:
        # In CI a missing API is a failure, not a skip: otherwise the pipeline goes green without testing.
        message = f"API not reachable at {API_URL} ({exc.__class__.__name__}). Start it first."
        pytest.fail(message) if os.getenv("CI") else pytest.skip(message)
    if not API_KEY:
        pytest.fail("Set API_KEY to the key the API was started with")
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        yield client


def test_health_needs_no_key(api):
    r = api.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str) and body["model_version"]


def test_predict_returns_the_contract_fields(api, healthy_run):
    r = api.post("/predict", json=healthy_run, headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == 0.5
    assert body["model_version"] == api.get("/health").json()["model_version"]


def test_predict_without_key_is_refused(api, healthy_run):
    assert api.post("/predict", json=healthy_run).status_code == 401


def test_predict_with_wrong_key_is_refused(api, healthy_run):
    # TODO (Lab 1, step 5): send the request with a wrong key and assert the status code is 401.
    pytest.skip("TODO: write this test")


def test_celsius_input_is_rejected(api, healthy_run):
    bad = {**healthy_run, "air_temp_k": 25.0}
    r = api.post("/predict", json=bad, headers={"X-API-Key": API_KEY})
    assert r.status_code == 422


def test_unknown_type_is_rejected(api, healthy_run):
    r = api.post("/predict", json={**healthy_run, "type": "X"}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 422


def test_prediction_is_logged(api, healthy_run):
    if not PREDICTION_LOG.exists():
        pytest.skip(f"{PREDICTION_LOG} is not on this machine (remote API?)")

    def count():
        with sqlite3.connect(PREDICTION_LOG) as con:
            return con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    before = count()
    api.post("/predict", json=healthy_run, headers={"X-API-Key": API_KEY})
    assert count() == before + 1
