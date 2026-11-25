"""Download the datasets, preprocess every image once, and write a manifest.

    python scripts/prepare_data.py            # downloads into data/raw, caches into data/cache

Outputs
-------
data/manifest.csv   one row per image: id, dataset, label, split, raw_path
data/cache/<view>/<id>.png   preprocessed images (view = full | disc) [^7]

Datasets [^2] [^3]
--------
* ORIGA (650 images, 168 glaucoma / 482 normal) - split 70/15/15 into
  train / val / test with a fixed seed, stratified by label [^4].
* VEIRC  (Venu Eye Institute, New Delhi) - 32 glaucoma / 225 normal / 107
  suspect. Never used for training or model selection: it is the *external*
  test set [^3]. "Suspect" images are kept in a separate split so we can see
  where the model places them, but they are excluded from binary metrics.

Image data is NOT redistributed in this repo (both datasets are licensed for
research use only); this script fetches it from the public mirrors [^2].
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nethralaya.preprocess import preprocess  # noqa: E402

ORIGA_REPO = "https://github.com/Quera-fr/Glaucoma-Dataset"  # mirror of ORIGA [^2]
VEIRC_REPO = "https://github.com/ProfMKD/Glaucoma-dataset"   # official release by the authors [^3]
CACHE_SIZE = 256  # cached side length; training crops 224 out of this [^9]
SEED = 42


def clone(url: str, dest: Path) -> None:
    if dest.exists():
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def extract_veirc(repo: Path, out: Path) -> None:
    if out.exists():
        return
    import py7zr  # only needed for this dataset's .7z archives

    zipfile.ZipFile(repo / "Glaucoma.zip").extractall(out / "glaucoma")
    for archive in repo.glob("*.7z"):
        cls = "suspect" if "uspect" in archive.name else "normal"
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(out / cls)


def origa_rows(root: Path) -> list[dict]:
    rows = []
    for split_dir in ("Train", "Validation"):  # the mirror's own split is ignored [^4]
        for p in sorted((root / split_dir).glob("*.jpg")):
            label = 1 if p.name.startswith("positive") else 0
            rows.append({"id": f"origa_{split_dir[0]}_{p.stem}", "dataset": "origa",
                         "label": label, "raw_path": str(p)})
    return rows


def veirc_rows(root: Path) -> list[dict]:
    rows = []
    for cls, label in (("glaucoma", 1), ("normal", 0), ("suspect", -1)):
        for i, p in enumerate(sorted((root / cls).rglob("*.jpg"))):
            rows.append({"id": f"veirc_{cls}_{i:03d}", "dataset": "veirc",
                         "label": label, "raw_path": str(p)})
    return rows


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 on ORIGA; VEIRC is external-only [^4]."""
    df = df.copy()
    origa = df[df.dataset == "origa"]
    train, rest = train_test_split(origa.index, test_size=0.30, stratify=origa.label,
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=origa.loc[rest, "label"],
                                 random_state=SEED)
    df.loc[train, "split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    veirc = df.dataset == "veirc"
    df.loc[veirc, "split"] = np.where(df.loc[veirc, "label"] == -1, "external_suspect", "external")
    return df


def _cache_one(args: tuple[str, str, Path]) -> None:
    raw_path, img_id, cache = args
    bgr = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read {raw_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR [^8]
    for view in ("full", "disc"):
        out = cache / view / f"{img_id}.png"
        if not out.exists():
            img = preprocess(rgb, view, CACHE_SIZE)
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    raw = args.data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    clone(ORIGA_REPO, raw / "origa")
    clone(VEIRC_REPO, raw / "veirc_repo")
    extract_veirc(raw / "veirc_repo", raw / "veirc")

    df = pd.DataFrame(origa_rows(raw / "origa") + veirc_rows(raw / "veirc"))
    df = assign_splits(df)

    cache = args.data / "cache"
    for view in ("full", "disc"):
        (cache / view).mkdir(parents=True, exist_ok=True)
    jobs = [(r.raw_path, r.id, cache) for r in df.itertuples()]
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_cache_one, jobs, chunksize=8))

    df.to_csv(args.data / "manifest.csv", index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
