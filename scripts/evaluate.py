"""Evaluate every trained run, pick the deployed model, and write the reports.

    python scripts/evaluate.py

* Internal test  = ORIGA held-out 15% (never seen during training/selection).
* External test  = VEIRC (different hospital, country and camera) [^3].
* The deployed view (disc vs full) is chosen on **validation** AUC, not on
  test, so the test numbers stay unbiased [^24].

Outputs: reports/metrics.json, reports/roc.png, reports/gradcam_examples.png,
and weights/glaucoma_effb0.pt (copy of the selected checkpoint). The web app
reads metrics.json, so the numbers it displays are the ones computed here [^27].
Footnote numbers refer to NOTES.md.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nethralaya.inference import GlaucomaPredictor  # noqa: E402
from nethralaya.metrics import calibrate, ece, sigmoid, summarise  # noqa: E402


def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def batched_logits(pred: GlaucomaPredictor, paths: list[str], bs: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(paths), bs):
        out.append(pred.logits([load_rgb(p) for p in paths[i:i + bs]]))
    return np.concatenate(out)


def main() -> None:
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    splits = ["val", "test", "external", "external_suspect"]

    results, probs = {}, {}
    for ckpt in sorted((ROOT / "runs").glob("*/model.pt")):
        pred = GlaucomaPredictor(ckpt)
        view, meta = pred.view, pred.meta
        a, b, thr = meta["platt_a"], meta["platt_b"], meta["threshold"]
        res = {"meta": meta}
        for split in splits:
            sub = manifest[manifest.split == split]
            # Re-run the *raw* images through the same inference path the app
            # uses, rather than reading the training cache [^8].
            lg = batched_logits(pred, sub.raw_path.tolist())
            p = calibrate(lg, a, b)
            probs[(view, split)] = (sub, p)
            if split == "external_suspect":
                # Clinically borderline (vertical CDR 0.4-0.7): no ground truth
                # for a binary metric; report where the model places them [^3].
                res[split] = {"n": int(len(p)), "median_prob": float(np.median(p)),
                              "frac_referred": float((p >= thr).mean())}
                continue
            y = sub.label.to_numpy()
            res[split] = summarise(y, p, thr)
            res[split]["ece_uncalibrated"] = ece(y, sigmoid(lg))  # before Platt scaling [^19]
        results[view] = res
        print(f"[{view}] " + "  ".join(
            f"{s}: AUC {res[s]['auc']:.3f}" for s in ("val", "test", "external")))

    # Model selection on validation AUC only [^24].
    chosen = max(results, key=lambda v: results[v]["val"]["auc"])
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs" / chosen / "model.pt", ROOT / "weights" / "glaucoma_effb0.pt")
    json.dump({"deployed_view": chosen, "views": results}, open(reports / "metrics.json", "w"), indent=2)

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split, title in zip(axes, ("test", "external"),
                                ("Internal test (ORIGA)", "External test (VEIRC, India)")):
        for view in results:
            sub, p = probs[(view, split)]
            fpr, tpr, _ = roc_curve(sub.label, p)
            r = results[view][split]
            ax.plot(fpr, tpr, label=f"{view}: AUC {r['auc']:.3f} "
                                    f"[{r['auc_ci95'][0]:.2f}-{r['auc_ci95'][1]:.2f}]")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(title=title, xlabel="1 - specificity", ylabel="sensitivity")
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(reports / "roc.png", dpi=130)

    # Learning curves (runs/ is git-ignored, so copy the histories into reports/) [^15]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for view, color in zip(results, ("tab:blue", "tab:orange")):
        h = pd.read_csv(ROOT / "runs" / view / "history.csv")
        h.to_csv(reports / f"history_{view}.csv", index=False)
        axes[0].plot(h.epoch, h.train_loss, color=color, label=f"{view} train")
        axes[0].plot(h.epoch, h.val_loss, "--", color=color, label=f"{view} val")
        axes[1].plot(h.epoch, h.val_auc, color=color, label=view)
        axes[1].axvline(results[view]["meta"]["best_epoch"], ls=":", color=color)
    axes[0].set(title="Loss (pos-weighted BCE)", xlabel="epoch"); axes[0].legend(fontsize=8)
    axes[1].set(title="Validation AUC (dotted = selected epoch)", xlabel="epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(reports / "learning_curves.png", dpi=130)

    # Grad-CAM examples from the internal test set (deployed model only)
    pred = GlaucomaPredictor(ROOT / "weights" / "glaucoma_effb0.pt")
    sub, p = probs[(chosen, "test")]
    sub = sub.assign(p=p)
    picks = pd.concat([sub[sub.label == 1].nlargest(3, "p"), sub[sub.label == 0].nsmallest(3, "p")])
    tiles = []
    for r in picks.itertuples():
        out = pred.predict(load_rgb(r.raw_path))
        tiles.append(np.vstack([out.model_input, out.heatmap]))
    grid = np.hstack(tiles)
    cv2.imwrite(str(reports / "gradcam_examples.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("deployed view:", chosen)


if __name__ == "__main__":
    main()
