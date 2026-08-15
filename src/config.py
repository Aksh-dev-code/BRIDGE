"""Shared paths and hyperparameters.

Everything the notebook hard-coded under /kaggle/working lives here instead,
so the same code runs locally. Any path can be overridden with an env var.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- data ---------------------------------------------------------------
# Extracted INCLUDE dataset: <DATA_ROOT>/<Category>/<Sign Name>/*.pkl
DATA_ROOT = Path(os.getenv("BRIDGE_DATA_ROOT", ROOT / "data" / "INCLUDE" / "Pose_Signs" / "Pose_Signs"))
# Where the prepared X.npy / y.npy / class_names.npy get cached
LANDMARKS_DIR = Path(os.getenv("BRIDGE_LANDMARKS_DIR", ROOT / "data" / "landmarks"))

X_PATH = LANDMARKS_DIR / "X.npy"
Y_PATH = LANDMARKS_DIR / "y.npy"
CLASS_NAMES_PATH = LANDMARKS_DIR / "class_names.npy"

# --- artifacts ----------------------------------------------------------
# Full bundle (weights + class names + config) — this is what `run` looks for.
MODEL_BUNDLE = Path(os.getenv("BRIDGE_MODEL", ROOT / "sign_model_full.pt"))
# Best-val checkpoint written during training (state_dict only).
BEST_CKPT = Path(os.getenv("BRIDGE_CKPT", ROOT / "sign_transformer_best.pt"))
RESULTS_DIR = Path(os.getenv("BRIDGE_RESULTS_DIR", ROOT / "results"))

# MediaPipe .task files used for live landmark extraction
MP_MODEL_DIR = Path(os.getenv("BRIDGE_MP_MODEL_DIR", ROOT / "models" / "mp_models"))
POSE_TASK = MP_MODEL_DIR / "pose_landmarker.task"
HAND_TASK = MP_MODEL_DIR / "hand_landmarker.task"

POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# --- feature layout -----------------------------------------------------
# INCLUDE Pose_Signs keypoints are MediaPipe Holistic ordered:
#   33 pose + 21 left hand + 21 right hand = 75 landmarks, (x, y, z) each.
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_LANDMARKS = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS  # 75
INPUT_DIM = NUM_LANDMARKS * 3  # 225

MAX_FRAMES = 40

# --- training defaults --------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
SEED = 42

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "max_len": MAX_FRAMES,
    "dropout": 0.2,
}


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
