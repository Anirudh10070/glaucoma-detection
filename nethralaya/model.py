"""Model construction and checkpoint I/O. Footnote numbers refer to NOTES.md."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import timm
import torch
import torch.nn as nn

ARCH = "efficientnet_b0"  # [^13]
# ImageNet weights published by timm's author as a GitHub release asset [^13].
PRETRAINED_URL = ("https://github.com/rwightman/pytorch-image-models/releases/download/"
                  "v0.1-weights/efficientnet_b0_ra-3dd342df.pth")


def build_model(pretrained_cache: Path | None = None) -> nn.Module:
    """EfficientNet-B0 with a single-logit head (binary task) [^14].

    ``drop_rate`` = dropout before the classifier, ``drop_path_rate`` =
    stochastic depth inside the MBConv blocks; both are regularisers that
    matter when you only have ~450 training images [^15].
    """
    model = timm.create_model(ARCH, pretrained=False, num_classes=1,
                              drop_rate=0.3, drop_path_rate=0.1)
    # timm's EfficientNet initialises Linear layers with U(-1/sqrt(fan_out), ..),
    # and fan_out = 1 for a single-logit head -> U(-1, 1) over 1280 features,
    # giving logits in the tens at step 0 (we saw a first-epoch loss of 4.4).
    # A small-normal init starts every prediction near p = 0.5 [^14].
    nn.init.normal_(model.classifier.weight, std=0.01)
    nn.init.zeros_(model.classifier.bias)
    if pretrained_cache is not None:
        pretrained_cache.parent.mkdir(parents=True, exist_ok=True)
        if not pretrained_cache.exists():
            urlretrieve(PRETRAINED_URL, pretrained_cache)
        state = torch.load(pretrained_cache, map_location="cpu", weights_only=True)
        # Drop the 1000-class ImageNet head; keep every feature-extractor weight.
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert set(missing) == {"classifier.weight", "classifier.bias"} and not unexpected, \
            (missing, unexpected)
    return model


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    """Weights + everything needed to reproduce inference (view, size, calibration, threshold) [^8]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    # weights_only=True refuses arbitrary pickled code - the original repo's
    # torch.load() would execute anything inside a malicious .pth [^16].
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(pretrained_cache=None)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["meta"]
