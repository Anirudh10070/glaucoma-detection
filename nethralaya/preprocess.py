"""Fundus image preprocessing.

Two deterministic steps run once, offline, before training (and again at
inference time inside the web app, so train and serve see identical inputs [^8]):

1. ``crop_fundus``  - remove the black camera border and pad to a square [^5].
2. ``locate_disc`` / ``crop_disc`` - find the optic disc (brightest blob) and
   cut a square region of interest around it [^6].

Footnote numbers refer to NOTES.md.
"""
from __future__ import annotations

import cv2
import numpy as np

# Pixels darker than this (0-255, max over RGB) are treated as camera border [^5].
BORDER_THRESHOLD = 20
# Side of the disc crop as a fraction of the fundus diameter [^6].
DISC_CROP_FRACTION = 0.40


def crop_fundus(rgb: np.ndarray) -> np.ndarray:
    """Crop to the circular field of view and zero-pad to a square [^5]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an RGB image of shape (H, W, 3)")
    mask = (rgb.max(axis=2) > BORDER_THRESHOLD).astype(np.uint8)
    # Opening removes isolated bright specks (text overlays, noise) in the border.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Guard against a failed threshold (e.g. a very dark image): only crop
        # if the detected field of view is a plausible size.
        if w * h > 0.2 * rgb.shape[0] * rgb.shape[1]:
            rgb = rgb[y:y + h, x:x + w]
    h, w = rgb.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=rgb.dtype)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = rgb
    return out


def locate_disc(fundus: np.ndarray, work_size: int = 512) -> tuple[float, float]:
    """Return the optic-disc centre as (x, y) in *relative* coordinates [0, 1] [^6].

    The disc is a bright, roughly disc-sized blob. A Difference-of-Gaussians
    (DoG) filter - blur at the disc's scale minus blur at a much larger scale -
    responds to exactly that: it ignores single bright pixels (too small) and
    broad illumination gradients such as flash glare on the FOV rim (too large).
    A plain "brightest pixel after blurring" version failed on rim glare; DoG
    fixed those cases on visual inspection of ~250 images [^6].
    """
    small = cv2.resize(fundus, (work_size, work_size), interpolation=cv2.INTER_AREA)
    # Red + green carry the disc contrast; blue is mostly noise in fundus photos.
    lum = 0.5 * small[..., 0].astype(np.float32) + 0.5 * small[..., 1].astype(np.float32)
    fov = (small.max(axis=2) > BORDER_THRESHOLD).astype(np.uint8)
    # Fill the black border with the mean retina brightness so the FOV edge
    # itself does not produce a strong DoG response.
    lum[fov == 0] = lum[fov == 1].mean() if fov.any() else 0
    dog = (cv2.GaussianBlur(lum, (0, 0), work_size / 50)
           - cv2.GaussianBlur(lum, (0, 0), work_size / 10))
    fov = cv2.erode(fov, np.ones((15, 15), np.uint8))
    dog[fov == 0] = -np.inf
    y, x = np.unravel_index(int(np.argmax(dog)), dog.shape)
    return x / work_size, y / work_size


def crop_disc(fundus: np.ndarray, fraction: float = DISC_CROP_FRACTION) -> np.ndarray:
    """Square crop centred on the detected optic disc [^6]."""
    side_px = fundus.shape[0]
    cx, cy = locate_disc(fundus)
    half = int(fraction * side_px / 2)
    cx, cy = int(cx * side_px), int(cy * side_px)
    # Clamp the window inside the image instead of padding, so the crop always
    # has real pixels and the disc stays roughly central.
    x0 = min(max(cx - half, 0), side_px - 2 * half)
    y0 = min(max(cy - half, 0), side_px - 2 * half)
    return fundus[y0:y0 + 2 * half, x0:x0 + 2 * half]


def preprocess(rgb: np.ndarray, view: str, out_size: int) -> np.ndarray:
    """Full offline pipeline: ``view`` is ``"full"`` or ``"disc"`` [^7]."""
    fundus = crop_fundus(rgb)
    img = crop_disc(fundus) if view == "disc" else fundus
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
