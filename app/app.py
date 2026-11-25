"""Nethralaya web demo.

    python app/app.py            # http://127.0.0.1:5000

Design rules (each one fixes a problem in the original repo):
* uploads are decoded in memory and never written to disk [^26]
* every number on the page comes from reports/metrics.json, produced by
  scripts/evaluate.py - nothing is hard-coded [^27]
* debug mode is off and uploads are size/type-limited [^28]
Footnote numbers refer to NOTES.md.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nethralaya.inference import GlaucomaPredictor  # noqa: E402

WEIGHTS = Path(os.environ.get("NETHRALAYA_WEIGHTS", ROOT / "weights" / "glaucoma_effb0.pt"))
METRICS = ROOT / "reports" / "metrics.json"
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB; Flask returns 413 above this [^28]

predictor = GlaucomaPredictor(WEIGHTS)
_report = json.load(open(METRICS))
REPORT = _report["views"][predictor.view]   # metrics of the model actually being served


def to_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def read_upload():
    f = request.files.get("image")
    if f is None or f.filename == "":
        return None, ("No file uploaded.", 400)
    if Path(f.filename).suffix.lower() not in ALLOWED:
        return None, ("Unsupported file type; upload a JPG/PNG/TIFF fundus photo.", 400)
    try:
        return predictor.decode(f.read()), None
    except ValueError:
        return None, ("Could not decode the image.", 400)


@app.get("/")
def index():
    return render_template("index.html", report=REPORT, view=predictor.view)


@app.post("/detect")
def detect():
    rgb, err = read_upload()
    if err:
        return render_template("index.html", report=REPORT, view=predictor.view, error=err[0]), err[1]
    out = predictor.predict(rgb)
    return render_template(
        "result.html",
        prob=out.probability, referable=out.referable, threshold=out.threshold,
        model_input=to_data_url(out.model_input), heatmap=to_data_url(out.heatmap),
        report=REPORT, view=predictor.view,
    )


@app.post("/api/predict")
def api_predict():
    """JSON endpoint for programmatic use / batch screening [^29]."""
    rgb, err = read_upload()
    if err:
        return jsonify(error=err[0]), err[1]
    out = predictor.predict(rgb, explain=False)
    return jsonify(probability=round(out.probability, 4), referable=out.referable,
                   threshold=round(out.threshold, 4), model_view=predictor.view)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
