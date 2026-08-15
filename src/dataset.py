"""INCLUDE Pose_Signs -> (X, y, class_names) arrays and torch DataLoaders.

Ported from notebook Steps 2-3. Prepared arrays are cached in
config.LANDMARKS_DIR so training doesn't re-walk the dataset every run.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config


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
    """One .pkl -> (max_frames, 225) float32, normalized and zero-padded."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    kp = d["keypoints"].astype(np.float32)  # (frames, 75, 3)
    h, w = d["vid_shape"][:2]

    # Normalize if values look like pixel coordinates
    if kp[..., 0].max() > 5:  # heuristic: normalized coords should be roughly 0-1
        kp[..., 0] /= w
        kp[..., 1] /= h
        # z left as-is (relative depth, no consistent pixel scale to divide by)

    seq = kp.reshape(kp.shape[0], -1)  # (frames, 225)

    if len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq[:max_frames]


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

    X_list, y_list = [], []
    for sign_name, sign_path in class_dirs:
        pkls = [f for f in os.listdir(sign_path) if f.endswith(".pkl")]
        for pkl_file in pkls:
            seq = load_pkl_sequence(os.path.join(sign_path, pkl_file), max_frames)
            X_list.append(seq)
            y_list.append(class_to_idx[sign_name])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "class_names.npy"), np.array(class_names))

    print("Final shapes:", X.shape, y.shape, "| classes:", len(class_names))
    return X, y, np.array(class_names)


def load_arrays(rebuild=False):
    """Return cached (X, y, class_names), building them first if needed."""
    cached = config.X_PATH.exists() and config.Y_PATH.exists() and config.CLASS_NAMES_PATH.exists()
    if rebuild or not cached:
        return build_arrays()

    X = np.load(config.X_PATH)
    y = np.load(config.Y_PATH)
    class_names = np.load(config.CLASS_NAMES_PATH)
    print(f"Loaded cached arrays: {X.shape} | classes: {len(class_names)}")
    return X, y, class_names


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


def make_loaders(X, y, batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT, seed=config.SEED):
    """Stratified train/val split -> (train_loader, val_loader) — notebook Step 3."""
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=val_split,
        stratify=y,
        random_state=seed,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print("Train:", X_train.shape, "Val:", X_val.shape)

    train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader
