"""Evaluation metrics. Footnote numbers refer to NOTES.md."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def sens_at_spec(y: np.ndarray, p: np.ndarray, target_spec: float = 0.85) -> float:
    """Sensitivity at a fixed specificity - the RIM-ONE DL benchmark's metric [^17]."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target_spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target_sens: float = 0.85) -> float:
    """Highest threshold that still reaches ``target_sens`` on this set [^18].

    Highest = fewest false positives among thresholds meeting the target.
    """
    for thr in np.sort(np.unique(p))[::-1]:
        if sens_spec(y, p, thr)[0] >= target_sens:
            return float(thr)
    return float(p.min())


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: |confidence - accuracy| averaged over bins [^19]."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: p = sigmoid(a * logit + b), (a, b) minimise validation NLL [^19].

    Plain temperature scaling (b = 0) is not enough here: training with
    ``pos_weight`` ~2.9 deliberately pushes every logit up by roughly
    log(2.9), so probabilities are biased towards "glaucoma". The bias term b
    removes that shift; a > 0 keeps the ranking, so AUC is unchanged.
    """
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # a = exp(log_a) > 0
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_a, b], lr=0.1, max_iter=500)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(lg * log_a.exp() + b, yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_a.exp().item()), float(b.item())


def calibrate(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a * logits + b)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI; resamples images with replacement [^20]."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():  # need both classes for AUC
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    sens, spec = sens_spec(y, p, thr)
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_ci(y, p, roc_auc_score)
    pred = p >= thr
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "auc": float(auc), "auc_ci95": [lo, hi],
        "sens_at_spec85": sens_at_spec(y, p, 0.85),
        "threshold": float(thr), "sensitivity": float(sens), "specificity": float(spec),
        "accuracy": float((pred == (y == 1)).mean()),
        "confusion": {"tp": int((pred & (y == 1)).sum()), "fn": int((~pred & (y == 1)).sum()),
                      "tn": int((~pred & (y == 0)).sum()), "fp": int((pred & (y == 0)).sum())},
        "ece": ece(y, p),
        "median_prob": {"glaucoma": float(np.median(p[y == 1])), "normal": float(np.median(p[y == 0]))},
    }
