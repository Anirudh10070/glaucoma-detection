"""Unit tests that run without any dataset [^30]. Footnote numbers refer to NOTES.md."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nethralaya.metrics import (calibrate, ece, fit_platt, sens_spec,  # noqa: E402
                                threshold_for_sensitivity)
from nethralaya.preprocess import crop_disc, crop_fundus, locate_disc, preprocess  # noqa: E402


def synthetic_fundus(w=1200, h=800, disc=(0.62, 0.45)):
    """Black canvas, orange circular retina, bright 'disc', plus rim glare."""
    img = np.zeros((h, w, 3), np.uint8)
    c, r = (w // 2, h // 2), int(h * 0.48)
    cv2.circle(img, c, r, (190, 90, 40), -1)
    cv2.circle(img, c, r, (240, 220, 140), 18)          # bright rim glare (a known failure mode)
    dx = int(c[0] - r + disc[0] * 2 * r); dy = int(c[1] - r + disc[1] * 2 * r)
    cv2.circle(img, (dx, dy), int(r * 0.09), (255, 230, 170), -1)
    return img


def test_crop_fundus_is_square_and_tight():
    out = crop_fundus(synthetic_fundus())
    assert out.shape[0] == out.shape[1]
    assert abs(out.shape[0] - 2 * int(800 * 0.48)) < 30   # cropped to the retina, not the canvas


def test_locate_disc_ignores_rim_glare():
    fundus = crop_fundus(synthetic_fundus(disc=(0.62, 0.45)))
    x, y = locate_disc(fundus)
    assert abs(x - 0.62) < 0.05 and abs(y - 0.45) < 0.05


def test_crop_disc_contains_disc_and_stays_inside():
    fundus = crop_fundus(synthetic_fundus(disc=(0.95, 0.5)))   # disc near the edge
    crop = crop_disc(fundus)
    assert crop.shape[0] == crop.shape[1] > 0
    assert crop.max() >= 250   # the bright disc is inside the crop


@pytest.mark.parametrize("view", ["full", "disc"])
def test_preprocess_output_size(view):
    assert preprocess(synthetic_fundus(), view, 256).shape == (256, 256, 3)


def test_sens_spec_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    assert sens_spec(y, p, 0.5) == (2 / 3, 2 / 3)
    thr = threshold_for_sensitivity(y, p, 1.0)
    assert thr == 0.3 and sens_spec(y, p, thr)[0] == 1.0


def test_platt_recovers_known_calibration():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < calibrate(logits, 0.5, -1.0)).astype(float)
    a, b = fit_platt(logits, y)
    assert abs(a - 0.5) < 0.05 and abs(b + 1.0) < 0.1


def test_ece_zero_for_perfect_calibration():
    p = np.repeat([0.2, 0.8], 1000)
    y = np.concatenate([np.r_[np.ones(200), np.zeros(800)], np.r_[np.ones(800), np.zeros(200)]])
    assert ece(y, p) < 1e-9
