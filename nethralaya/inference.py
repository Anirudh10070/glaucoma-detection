"""Single entry point for inference, shared by the web app and evaluate.py so
there is exactly one code path from raw image to probability [^8].
Footnote numbers refer to NOTES.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import eval_transform
from .gradcam import grad_cam, overlay
from .metrics import calibrate
from .model import load_checkpoint
from .preprocess import preprocess

CACHE_SIZE = 256  # must equal scripts/prepare_data.py CACHE_SIZE (train/serve parity) [^8]


@dataclass
class Prediction:
    probability: float      # calibrated P(glaucoma) [^19]
    referable: bool         # probability >= operating threshold [^18]
    threshold: float
    model_input: np.ndarray  # RGB uint8, what the network actually saw
    heatmap: np.ndarray      # RGB uint8 Grad-CAM overlay [^25]


class GlaucomaPredictor:
    def __init__(self, checkpoint: Path):
        self.model, self.meta = load_checkpoint(checkpoint)
        self.transform = eval_transform()
        self.view = self.meta["view"]

    def decode(self, data: bytes) -> np.ndarray:
        """Bytes -> RGB array, entirely in memory (no temp files) [^26]."""
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not a decodable image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def logits(self, rgb_batch: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([self.transform(preprocess(r, self.view, CACHE_SIZE)) for r in rgb_batch])
        with torch.no_grad():
            return self.model(x).squeeze(1).numpy()

    def predict(self, rgb: np.ndarray, explain: bool = True) -> Prediction:
        cached = preprocess(rgb, self.view, CACHE_SIZE)
        x = self.transform(cached).unsqueeze(0)
        with torch.no_grad():
            logit = float(self.model(x).squeeze())
        prob = float(calibrate(np.array([logit]), self.meta["platt_a"], self.meta["platt_b"])[0])
        referable = prob >= self.meta["threshold"]
        shown = cv2.resize(cached, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_AREA)
        heat = shown
        if explain:
            cam = grad_cam(self.model, x, self.model.bn2, positive=referable)
            heat = overlay(shown, cam)
        return Prediction(prob, referable, float(self.meta["threshold"]), shown, heat)
