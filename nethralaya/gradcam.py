"""Grad-CAM for a single-logit classifier [^25]. Footnote numbers refer to NOTES.md."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def grad_cam(model: nn.Module, x: torch.Tensor, layer: nn.Module, positive: bool) -> np.ndarray:
    """Return a [0, 1] heat-map of shape (H, W) matching ``x``.

    positive=True explains the *glaucoma* logit; False explains the *normal*
    decision by back-propagating -logit, so the map always shows evidence for
    whatever the model predicted.
    """
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        store["act"] = out
        # Tensor hook instead of module.register_backward_hook (deprecated and
        # wrong for modules with several ops - the original repo used it) [^25].
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logit = model(x).squeeze()
        (logit if positive else -logit).backward()
    finally:
        handle.remove()

    act, grad = store["act"][0], store["grad"][0]          # (C, h, w)
    weights = grad.mean(dim=(1, 2))                         # global-average-pooled gradients
    cam = torch.relu((weights[:, None, None] * act).sum(0)).detach().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
    rng = cam.max() - cam.min()
    return (cam - cam.min()) / rng if rng > 0 else np.zeros_like(cam)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a JET heat-map onto an RGB image.

    cv2.applyColorMap returns **BGR**; converting it to RGB before blending is
    the colour bug the original repo had (its overlays were red/blue swapped) [^8].
    """
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)
