"""Fine-tune EfficientNet-B0 for glaucoma vs normal.

    python scripts/train.py --view disc          # optic-disc crop (default model)
    python scripts/train.py --view full          # whole fundus (ablation) [^7]

Writes runs/<view>/model.pt (weights + calibration + threshold) and
runs/<view>/history.csv. Footnote numbers refer to NOTES.md.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nethralaya.data import FundusDataset, eval_transform, train_transform  # noqa: E402
from nethralaya.metrics import calibrate, fit_platt, threshold_for_sensitivity  # noqa: E402
from nethralaya.model import ARCH, build_model, save_checkpoint  # noqa: E402

TARGET_SENSITIVITY = 0.85  # operating point chosen on validation [^18]


def seed_everything(seed: int) -> None:
    """Reproducibility [^21]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x).squeeze(1))
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["disc", "full"], default="disc")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)  # chosen on validation [^23]
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    out = args.out or ROOT / "runs" / args.view
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    manifest = pd.read_csv(args.data / "manifest.csv")
    cache = args.data / "cache"
    train_ds = FundusDataset(manifest[manifest.split == "train"], cache, args.view, train_transform())
    val_ds = FundusDataset(manifest[manifest.split == "val"], cache, args.view, eval_transform())
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = build_model(pretrained_cache=ROOT / ".cache" / "efficientnet_b0_imagenet.pth")

    # Class imbalance (~1 glaucoma : 2.9 normal): up-weight positive examples
    # in the loss instead of over-sampling [^22].
    n_pos = float(train_ds.labels.sum()); n_neg = len(train_ds) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))

    # AdamW + 1-epoch linear warm-up + cosine decay, stepped per batch [^23].
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_dl)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    best_auc, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
        val_logits, val_y = predict_logits(model, val_dl)
        val_auc = roc_auc_score(val_y, val_logits)
        val_loss = loss_fn(torch.tensor(val_logits), torch.tensor(val_y)).item()
        history.append({"epoch": epoch, "train_loss": running / (steps_per_epoch * args.batch),
                        "val_loss": val_loss, "val_auc": val_auc, "secs": time.time() - t0})
        print(f"epoch {epoch:2d}  train_loss {history[-1]['train_loss']:.4f}  "
              f"val_loss {val_loss:.4f}  val_auc {val_auc:.4f}  ({history[-1]['secs']:.0f}s)", flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        # Model selection on validation AUC, never on test [^24].
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), out / "best_weights.pt")
        elif epoch - best_epoch >= args.patience:  # early stopping [^24]
            print(f"early stop: no val AUC improvement for {args.patience} epochs")
            break

    # Post-hoc calibration + operating point, both fitted on validation only.
    model.load_state_dict(torch.load(out / "best_weights.pt", weights_only=True))
    val_logits, val_y = predict_logits(model, val_dl)
    platt_a, platt_b = fit_platt(val_logits, val_y)                        # [^19]
    val_prob = calibrate(val_logits, platt_a, platt_b)
    threshold = threshold_for_sensitivity(val_y, val_prob, TARGET_SENSITIVITY)  # [^18]

    meta = {"arch": ARCH, "view": args.view, "img_size": 224, "platt_a": platt_a, "platt_b": platt_b,
            "threshold": threshold, "target_sensitivity": TARGET_SENSITIVITY,
            "best_epoch": best_epoch, "val_auc": best_auc, "seed": args.seed,
            "train_counts": {"pos": int(n_pos), "neg": int(n_neg)}}
    save_checkpoint(out / "model.pt", model, meta)
    (out / "best_weights.pt").unlink()
    print("saved", out / "model.pt", meta)


if __name__ == "__main__":
    main()
