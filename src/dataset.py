"""INCLUDE Pose_Signs -> (X, y, class_names) arrays and torch DataLoaders.

Ported from notebook Steps 2-3. Prepared arrays are cached in
config.LANDMARKS_DIR so training doesn't re-walk the dataset every run.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config, preprocess


def find_class_dirs(data_root=None):
    """List (sign_name, path) for every non-empty sign folder."""
    data_root = str(data_root or config.DATA_ROOT)
    class_dirs = []
    for category in sorted(os.listdir(data_root)):
        category_path = os.path.join(data_root, category)
        if not os.path.isdir(category_path):
            continue
        for sign_name in sorted(os.listdir(category_path)):
            sign_path = os.path.join(category_path, sign_name)
            if not os.path.isdir(sign_path):
                continue
            n_pkls = len([f for f in os.listdir(sign_path) if f.endswith(".pkl")])
            if n_pkls > 0:
                class_dirs.append((sign_name, sign_path))
    return class_dirs


def load_pkl_sequence(pkl_path, max_frames=config.MAX_FRAMES):
    """One .pkl -> (max_frames, 225) float32, or None if the clip is unusable.

    The actual work is in preprocess.encode, which the live camera path calls
    too — that shared call is what keeps training and inference in step.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return preprocess.encode(
        d["keypoints"],
        d.get("confidences"),          # flags the landmarks MediaPipe guessed at
        vid_shape=d["vid_shape"],
        max_frames=max_frames,
    )


def build_arrays(data_root=None, out_dir=None, max_frames=config.MAX_FRAMES):
    """Walk the dataset, encode every clip, and cache X/y/class_names to disk."""
    data_root = data_root or config.DATA_ROOT
    out_dir = out_dir or config.LANDMARKS_DIR

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset not found at {data_root}.\n"
            "Download and extract INCLUDE (Pose_Signs) there, or point\n"
            "BRIDGE_DATA_ROOT at your copy. See README for the download steps."
        )

    class_dirs = find_class_dirs(data_root)
    class_names = sorted(set(name for name, _ in class_dirs))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"Usable classes (non-empty): {len(class_names)}")

    X_list, y_list, id_list, skipped = [], [], [], []
    for sign_name, sign_path in class_dirs:
        pkls = [f for f in os.listdir(sign_path) if f.endswith(".pkl")]
        for pkl_file in pkls:
            path = os.path.join(sign_path, pkl_file)
            seq = load_pkl_sequence(path, max_frames)
            if seq is None:
                # No frame had both shoulders, so there is no reference length
                # to normalize against — the clip can't be encoded.
                skipped.append(path)
                continue
            X_list.append(seq)
            y_list.append(class_to_idx[sign_name])
            id_list.append(os.path.relpath(path, data_root))

    if skipped:
        print(f"Skipped {len(skipped)} unusable clips (no shoulders detected), e.g.:")
        for p in skipped[:3]:
            print(f"  {p}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "class_names.npy"), np.array(class_names))
    np.save(os.path.join(out_dir, "sample_ids.npy"), np.array(id_list))
    np.save(os.path.join(out_dir, "version.npy"), np.array([config.PREPROCESS_VERSION]))

    print("Final shapes:", X.shape, y.shape, "| classes:", len(class_names))
    return X, y, np.array(class_names)


def load_arrays(rebuild=False):
    """Return cached (X, y, class_names), building them first if needed.

    The cache is rebuilt automatically when it was written by an older
    preprocessing version — stale arrays are indistinguishable from fresh ones
    once loaded, so this can't be left to whoever remembers --rebuild-data.
    """
    cached = config.X_PATH.exists() and config.Y_PATH.exists() and config.CLASS_NAMES_PATH.exists()

    if cached and not rebuild:
        version_path = config.LANDMARKS_DIR / "version.npy"
        version = int(np.load(version_path)[0]) if version_path.exists() else 1
        if version != config.PREPROCESS_VERSION:
            print(f"Cached arrays are preprocessing v{version}, code is "
                  f"v{config.PREPROCESS_VERSION} — re-encoding the dataset.")
            rebuild = True

    if rebuild or not cached:
        return build_arrays()

    X = np.load(config.X_PATH)
    y = np.load(config.Y_PATH)
    class_names = np.load(config.CLASS_NAMES_PATH)
    print(f"Loaded cached arrays: {X.shape} | classes: {len(class_names)}")
    return X, y, class_names


def load_sample_ids():
    """Per-sample dataset-relative paths, or None if the cache predates them."""
    if not config.SAMPLE_IDS_PATH.exists():
        return None
    return np.load(config.SAMPLE_IDS_PATH)


def sanity_check(X, y):
    """Notebook's post-prep sanity check."""
    print("Any NaN or Inf?", np.isnan(X).any(), np.isinf(X).any())
    print("Value range:", X.min(), X.max())
    print("Fraction of zeroed (low-confidence/padding) values:", (X == 0).mean())

    _, counts = np.unique(y, return_counts=True)
    print(f"Samples per class — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
    print("Classes with fewer than 5 samples:", (counts < 5).sum())


class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(X, y, batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT,
                 seed=config.SEED, split="auto", sample_ids=None):
    """Train/val split -> (train_loader, val_loader).

    split="auto"     use INCLUDE's official signer-disjoint split if available,
                     otherwise fall back to the random stratified one
    split="official" require the official split, fail if it can't be used
    split="random"   always use the random stratified split
    """
    from sklearn.model_selection import train_test_split

    from src import splits as split_mod

    train_idx = val_idx = None
    if split in ("auto", "official"):
        if sample_ids is None:
            sample_ids = load_sample_ids()
        if sample_ids is None:
            msg = ("No sample_ids.npy in the cache — re-encode the dataset with "
                   "`--rebuild-data` to use the official split.")
            if split == "official":
                raise RuntimeError(msg)
            print(msg)
        else:
            official = split_mod.load_official_split(sample_ids)
            if official is not None:
                train_idx, val_idx = official
                print(f"Using INCLUDE's official signer-disjoint split "
                      f"({len(train_idx)} train / {len(val_idx)} val)")
            elif split == "official":
                raise RuntimeError(
                    "The official split was requested but could not be used — see "
                    "the messages above."
                )

    if train_idx is None:
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        singleton = counts[y] < 2
        idx = np.arange(len(y))

        if singleton.any():
            print(f"{singleton.sum()} sample(s) in {(counts == 1).sum()} single-example "
                  f"class(es) -> train only (can't be split)")
        train_idx, val_idx = train_test_split(
            idx[~singleton],
            test_size=val_split,
            stratify=y[~singleton],
            random_state=seed,
        )
        train_idx = np.concatenate([train_idx, idx[singleton]])
        print("Using a random stratified split — validation accuracy will be "
              "optimistic, because the same signer can appear on both sides.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print("Train:", X_train.shape, "Val:", X_val.shape)

    train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader
