"""INCLUDE's official train/test split.

`INCLUDE.zip` ships a `Train_Test_Split` archive listing which clip belongs to
which side. It matters because the official split is **signer-disjoint** — the
people in the test set never appear in training.

A random split, which is what this project used before, lets the same signer
land on both sides. The model then gets credit for recognising *people*, and
validation accuracy comes out higher than the model deserves. Numbers from the
two splits are not comparable, and only the official one is comparable to
published INCLUDE results.

The split tables index clips by video filename (`MVI_4512.MOV`), while the pose
dataset stores `MVI_4512.pkl`, so matching is done on the filename stem.
"""

import os
import re
from pathlib import Path

import numpy as np

from src import config

# Column names seen across INCLUDE's split tables; matched case-insensitively.
_PATH_COLUMNS = ("filepath", "file_path", "path", "video", "videopath", "filename", "file")
_WORD_COLUMNS = ("word", "gloss", "label", "sign", "class")

_SPLIT_DIR_HINTS = ("train_test_split", "train-test-split", "splits", "split")


def _normalize(text):
    return re.sub(r"\s+", " ", str(text).replace("\\", "/").strip().lower())


def path_key(path):
    """'Adjectives/4. sad/MVI_9568.MOV' -> 'adjectives/4. sad/mvi_9568'.

    The full relative path, because filenames alone are not unique: 508 stems
    are reused across classes, and matching on those puts the same clip in both
    halves of the split.
    """
    p = _normalize(path)
    return os.path.splitext(p)[0]


def stem_key(path):
    """'Adjectives/4. sad/MVI_9568.MOV' -> 'mvi_9568' — the ambiguous fallback."""
    return os.path.splitext(_normalize(Path(_normalize(path)).name))[0]


def _looks_like_split_dir(path):
    return path.is_dir() and any(h in path.name.lower() for h in _SPLIT_DIR_HINTS)


def find_split_dir(start=None):
    """Look around the dataset for the extracted Train_Test_Split folder.

    Unzipping INCLUDE.zip leaves the split beside a *sibling* of the pose data
    (`INCLUDE/INCLUDE/Train_Test_Split` next to `INCLUDE/Pose_Signs`), so each
    ancestor's grandchildren get checked too, not just its direct children.
    """
    start = Path(start or config.DATA_ROOT).resolve()
    for base in [start, *start.parents][:6]:
        if not base.exists():
            continue
        try:
            children = [e for e in base.iterdir() if e.is_dir()]
        except OSError:
            continue
        for child in children:
            if _looks_like_split_dir(child):
                return child
        for child in children:
            try:
                for grandchild in child.iterdir():
                    if _looks_like_split_dir(grandchild):
                        return grandchild
            except OSError:
                continue
    return None


def find_split_tables(split_dir=None):
    """-> {'train': Path, 'test': Path} for whichever tables exist."""
    split_dir = split_dir or find_split_dir()
    if split_dir is None:
        return {}

    tables = {}
    for path in sorted(Path(split_dir).rglob("*")):
        if path.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        name = path.name.lower()
        if "train" in name:
            tables.setdefault("train", path)
        elif "test" in name or "val" in name:
            tables.setdefault("test", path)
    return tables


def _read_table(path):
    import pandas as pd

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_excel(path)
    except ImportError as e:
        raise RuntimeError(
            f"{path.name} is a spreadsheet and pandas has no engine for it here.\n"
            f"Install one with `uv add openpyxl`, or convert it to CSV.\n({e})"
        ) from None


def _pick_column(df, candidates):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    return None


def read_split_paths(path):
    """Read one split table -> the list of clip paths it names."""
    df = _read_table(path)
    col = _pick_column(df, _PATH_COLUMNS)

    if col is None:
        # Some releases have no header row, just one path per line.
        if df.shape[1] == 1:
            col = df.columns[0]
        else:
            raise RuntimeError(
                f"{path.name} has columns {list(df.columns)[:8]} and none look like a "
                "file path. Tell me the right column and I'll wire it in."
            )
    return [str(v) for v in df[col].dropna()]


def load_official_split(sample_ids, verbose=True):
    """sample_ids: per-sample 'Category/Sign/file.pkl' -> (train_idx, val_idx) or None.

    Returns None — rather than raising — when the split files aren't present or
    don't line up with the clips on disk, so training falls back to the random
    split instead of failing.
    """
    tables = find_split_tables()
    if not tables:
        if verbose:
            print("No Train_Test_Split tables found — using the random split.")
            print("  (unzip INCLUDE/INCLUDE/Train_Test_Split.zip to enable the official one)")
        return None

    if verbose:
        for name, path in tables.items():
            print(f"Official split table [{name}]: {path}")

    try:
        train_paths = read_split_paths(tables["train"]) if "train" in tables else []
        test_paths = read_split_paths(tables["test"]) if "test" in tables else []
    except RuntimeError as e:
        print(f"Couldn't read the split tables ({e}) — using the random split.")
        return None

    if not train_paths and not test_paths:
        print("Split tables were empty — using the random split.")
        return None

    # Full relative path first; fall back to bare filenames only if the layouts
    # don't line up, since those are ambiguous across classes.
    in_train = in_test = None
    for keyfn, label in ((path_key, "full path"), (stem_key, "filename only")):
        train_keys = {keyfn(p) for p in train_paths}
        test_keys = {keyfn(p) for p in test_paths}
        keys = np.array([keyfn(s) for s in sample_ids])
        t = np.array([k in train_keys for k in keys])
        v = np.array([k in test_keys for k in keys])

        if (t | v).mean() >= 0.5:
            both = int((t & v).sum())
            if both:
                print(f"  {both} clip(s) matched BOTH halves by {label} — ambiguous, "
                      "treating them as test to avoid leaking them into training")
                t = t & ~v
            in_train, in_test = t, v
            if verbose:
                print(f"Matched by {label}: {(t | v).sum()}/{len(keys)} clips "
                      f"({t.sum()} train, {v.sum()} test)")
            break

    if in_train is None:
        print("The split tables don't line up with this dataset — using the random split.")
        return None
    matched = in_train | in_test

    # A near-total mismatch means the key scheme is wrong, and silently training
    # on whatever did match would be worse than not using the split at all.
    if matched.mean() < 0.5:
        print(f"Only {matched.mean():.0%} of clips matched — the split tables don't "
              "line up with this dataset. Using the random split instead.")
        return None
    if in_test.sum() == 0:
        print("No clips landed in the test half — using the random split instead.")
        return None

    if matched.sum() < len(keys):
        unmatched = int((~matched).sum())
        print(f"  {unmatched} clip(s) appear in neither table -> added to train")
        in_train = in_train | ~matched

    return np.flatnonzero(in_train), np.flatnonzero(in_test)
