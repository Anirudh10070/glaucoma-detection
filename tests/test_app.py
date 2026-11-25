"""End-to-end tests of the Flask app; skipped if the trained weights are absent."""
import io
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / "weights" / "glaucoma_effb0.pt").exists(),
                                reason="trained weights not present")


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(ROOT / "app"))
    import app as webapp  # noqa: WPS433
    return webapp.app.test_client()


def png_bytes():
    from test_core import synthetic_fundus
    ok, buf = cv2.imencode(".png", synthetic_fundus())
    return io.BytesIO(buf.tobytes())


def snapshot(root: Path):
    return {p for p in root.rglob("*") if ".git" not in p.parts and "__pycache__" not in p.parts}


def test_detect_returns_result_and_writes_nothing(client):
    before = snapshot(ROOT)
    r = client.post("/detect", data={"image": (png_bytes(), "eye.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"P(glaucoma)" in r.data and b"data:image/png;base64" in r.data
    assert snapshot(ROOT) == before          # privacy claim actually holds [^26]


def test_api_predict_json(client):
    r = client.post("/api/predict", data={"image": (png_bytes(), "eye.png")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["referable"], bool)


@pytest.mark.parametrize("payload,name", [(b"not an image", "x.png"), (b"abc", "x.exe")])
def test_rejects_bad_uploads(client, payload, name):
    r = client.post("/api/predict", data={"image": (io.BytesIO(payload), name)},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_metrics_on_page_come_from_report(client):
    import json
    rep = json.load(open(ROOT / "reports" / "metrics.json"))
    auc = rep["views"][rep["deployed_view"]]["test"]["auc"]
    assert f"{auc:.3f}".encode() in client.get("/").data   # no hard-coded numbers [^27]
