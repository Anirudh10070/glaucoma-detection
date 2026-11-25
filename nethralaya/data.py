"""Dataset + transforms. Footnote numbers refer to NOTES.md."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet statistics: the backbone was pretrained with these, so inputs must
# be normalised the same way (the original repo used 0.5/0.5, a mismatch) [^10].
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224  # EfficientNet-B0's native resolution [^9]


def train_transform() -> T.Compose:
    """Augmentations [^11]."""
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),          # left eye <-> right eye: anatomically valid
        T.RandomRotation(15),              # camera / head tilt
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),  # camera/flash variation
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> T.Compose:
    """Deterministic: resize the whole cached image, no crop [^9]."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FundusDataset(Dataset):
    """Reads preprocessed images from data/cache/<view>/ and keeps them in RAM.

    The whole cache is ~100 MB, so pre-loading removes disk/PNG-decode time from
    every epoch - this matters on a 2-core CPU [^12].
    """

    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, view: str, transform):
        self.ids = manifest["id"].tolist()
        self.labels = manifest["label"].astype(np.float32).to_numpy()
        self.transform = transform
        self.images = []
        for i in self.ids:
            bgr = cv2.imread(str(cache_dir / view / f"{i}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"missing cache image for {i}; run scripts/prepare_data.py")
            self.images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), torch.tensor(self.labels[idx])
