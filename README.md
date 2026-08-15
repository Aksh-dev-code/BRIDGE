# BRIDGE — Indian Sign Language Recognition

A transformer that classifies Indian Sign Language signs from body + hand
landmark sequences, trained on the [INCLUDE](https://zenodo.org/record/6674324)
dataset (Pose_Signs split).

The model logic is a direct port of `isl-transformer-openhand.ipynb` — the
notebook is kept as-is for reference; `src/` is the runnable version of it.

## Setup

```bash
uv sync
```

## The two commands

### Run the model

```bash
uv run python main.py run
```

Opens your camera and recognizes signs live. It first checks for a trained
model at `sign_model_full.pt` — **if there isn't one, it trains a model first,
then starts recognizing.**

```bash
# phone camera / IP stream instead of the built-in webcam
uv run python main.py run --source http://10.118.2.94:8080/video

# classify a recorded video instead of a live feed
uv run python main.py run --video clip.mp4

# fail instead of auto-training when no model exists
uv run python main.py run --no-train
```

While the window is open: **Q** or **ESC** quits, **R** resets the frame
window, **C** clears the recognized-sign history.

### How a sign gets recognized

Landmarks stream into a 40-frame window, but a sign is only **committed** once
the same label wins `--stability` windows in a row above `--threshold`. Two
things end a sign and clear the window, so one sign never bleeds into the next:

1. **Dropping your hands** for `--idle-reset` frames — the natural boundary
   between signs.
2. **Committing a sign** — the window restarts empty for the following one.

So the rhythm is: raise hands → sign → drop hands → sign the next one. The
overlay shows `READING n/40` while the window fills, `CONFIRMING n/5` while
windows are agreeing, and `IDLE` when your hands are down. Committed signs
print to the terminal and build a transcript along the bottom of the window.

Without this, the window slides across sign boundaries and roughly 3 in 4
windows end up containing two half-signs, which is exactly when the model
produces confident nonsense.

```bash
# commit faster, at the cost of more twitch
uv run python main.py run --stability 3

# require higher confidence before a window counts
uv run python main.py run --threshold 0.6

# you sign continuously without dropping your hands much
uv run python main.py run --idle-reset 4
```

Other flags: `--topk N` (how many live guesses to show), `--swap-hands` (if
left/right look mirrored), `--no-mirror` (don't flip the preview).

### Train the model

```bash
uv run python main.py train
```

Prepares the dataset (cached to `data/landmarks/`), trains for 100 epochs,
saves the best checkpoint to `sign_transformer_best.pt` and the full bundle
(weights + class names + config) to `sign_model_full.pt`, then evaluates and
shows the charts.

```bash
uv run python main.py train --epochs 50          # shorter run
uv run python main.py train --rebuild-data       # re-encode the .pkl dataset
uv run python main.py train --resume             # warm-start from the checkpoint
uv run python main.py train --no-eval            # skip the evaluation pass
uv run python main.py train --no-show            # save charts without opening them
```

### Bonus: evaluate an existing model

```bash
uv run python main.py eval
```

Scores the saved model and redraws the charts without retraining.

**`eval` needs the dataset**, because accuracy can only be measured against
labelled clips — the model file alone can't tell you how good it is. If you
trained somewhere else (Kaggle, a GPU box) and don't have the data locally,
copy `results/eval_cache.npz` back from that machine. Every `train` and `eval`
run writes it, and `eval` replays the numbers and regenerates every chart from
it offline. The summary says `(replayed from cache)` when that happens.

`run` needs no dataset at all.

## What you see at the end

Same output the notebook produces, just from the terminal — worst/best classes
by F1, top-1/3/5 accuracy, the 30 most-confused sign pairs, then a summary:

```
============================================================
PERFORMANCE SUMMARY
============================================================
  Validation accuracy : 84.21%
  Top-3 / Top-5       : 94.74% / 96.49%
  Classes             : 262  (485 val samples)
  Perfect classes     : 118 (F1 = 1.00)
  Failed classes      : 9 (F1 = 0.00)
  Mean confidence     : 0.81 when right, 0.44 when wrong
============================================================
```

Then the charts open on screen and are saved to `results/`:

| Chart | Shows |
| --- | --- |
| `training_curves.png` | loss + train/val accuracy per epoch (`train` only) |
| `confusion_pairs.png` | the 20 most common true → predicted mistakes |
| `worst_classes_f1.png` | the 30 weakest classes by F1 |
| `confidence_dist.png` | confidence when right vs when wrong |
| `confusion_matrix_top40_normalized.png` | heatmap of the 40 most-confused classes |

Windows open automatically when you're in a terminal that supports it; the run
finishes once you close them. On a headless machine (or piped output) the
charts are only written to `results/`. Force that with `--no-show`, or skip
plotting entirely with `--no-plots`.

## Getting the dataset (only needed for `train`)

```bash
mkdir -p data && cd data
wget -O INCLUDE.zip "https://zenodo.org/record/6674324/files/INCLUDE.zip?download=1"
unzip -q INCLUDE.zip -d INCLUDE
unzip -q INCLUDE/INCLUDE/Pose_Signs.zip -d INCLUDE/Pose_Signs
```

The expected layout is `data/INCLUDE/Pose_Signs/Pose_Signs/<Category>/<Sign>/*.pkl`.
Point somewhere else with `BRIDGE_DATA_ROOT=/path/to/Pose_Signs`.

## Layout

| Path | What it does |
| --- | --- |
| `main.py` | CLI — `train`, `run`, `eval` |
| `src/config.py` | Paths, hyperparameters, feature layout (env-overridable) |
| `src/dataset.py` | INCLUDE `.pkl` → `X`/`y` arrays, split, DataLoaders |
| `src/model.py` | `SignTransformer`, bundle save/load |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Metrics + plots into `results/` |
| `src/landmarks.py` | Live MediaPipe landmark extraction (75 × 3) |
| `src/infer.py` | Rolling-window live recognition, video inference |
| `src/Data Collection.py` | Standalone webcam capture script (unchanged) |

## How live inference matches training

INCLUDE Pose_Signs stores 75 landmarks per frame in MediaPipe Holistic order —
33 pose + 21 left hand + 21 right hand, `(x, y, z)` each = **225 features**.
`src/landmarks.py` rebuilds exactly that layout from the MediaPipe pose and
hand landmarker tasks, so a webcam frame becomes a vector the trained model
already understands. Sequences are 40 frames, zero-padded when short.

> MediaPipe is pinned to `<1.0` — the 1.x vision tasks crash on macOS
> (`Check failed: service_ Service is unavailable`).

## Environment overrides

`BRIDGE_DATA_ROOT`, `BRIDGE_LANDMARKS_DIR`, `BRIDGE_MODEL`, `BRIDGE_CKPT`,
`BRIDGE_RESULTS_DIR`, `BRIDGE_MP_MODEL_DIR`.

`BRIDGE_VERBOSE_MP=1` restores MediaPipe's raw console output. By default its
startup chatter and failed telemetry uploads (`Failed to send to clearcut`) are
filtered out — they come from MediaPipe's C++ layer writing straight to file
descriptor 2, which no logging env var can silence. Real errors and tracebacks
are never filtered.
