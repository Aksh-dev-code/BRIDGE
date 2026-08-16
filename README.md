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

## The commands

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

While the window is open: **Q** or **ESC** quits, **R** drops the gesture being
recorded, **C** clears the recognized-sign history, **D** toggles debug numbers.

### How a sign gets recognized

One sign at a time, segmented by your hands:

1. **Hands come up** — recording starts.
2. **You sign.**
3. **Hands go down** — after `--end-silence` seconds the gesture closes, and the
   whole thing is classified in one go.

The rhythm is: raise hands → sign → drop hands → sign the next one. The overlay
shows `READY` when it's waiting, `RECORDING 1.4s` while you sign, and
`FINISHING` once your hands drop. Committed signs print to the terminal and
build a transcript along the bottom of the window.

The reason for segmenting rather than sliding a window: training encodes a
*whole* clip, start of sign to end, resampled onto 40 frames. A window holding
half of one sign and half of the next is not something the model was ever shown,
so no amount of threshold tuning makes those predictions mean anything. Feeding
it complete gestures is the only way the numbers line up.

Brief detection dropouts are buffered through — MediaPipe losing a hand
mid-motion is a hiccup, not the end of a sign — and the trailing frames after
your hands drop are trimmed off before classifying.

```bash
# your signs are quicker or slower than the defaults
uv run python main.py run --min-duration 0.3 --max-duration 4

# commit sooner after your hands drop
uv run python main.py run --end-silence 0.3

# only accept confident predictions
uv run python main.py run --threshold 0.6 --margin 0.25

# show live fps / frame counts on the preview
uv run python main.py run --debug
```

`--threshold` needs calibrating against your own trained model: `eval` prints
`Mean confidence: X when right, Y when wrong`, and a value between those two is
the right starting point.

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

### Evaluate an existing model

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

### Check the features

```bash
uv run python main.py diagnose            # needs the dataset + a camera
uv run python main.py diagnose --no-camera
```

Walks a sample of the `.pkl` files, works out how they must be normalized, then
captures a few seconds from your webcam and prints the two feature distributions
side by side. Run it before a long training job — if training and live features
disagree, the model will look fine in `eval` and fall apart on camera, and this
is what shows that.

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

The archive is **663 MB**, and Zenodo redirects, so the download needs to follow
redirects (`-L`). The outer archive contains more archives, which is why it gets
unzipped twice.

### macOS / Linux

macOS has no `wget` unless you installed it yourself — `curl` and `unzip` are
both built in, so this needs nothing extra:

```bash
mkdir -p data && cd data

# -L follows Zenodo's redirect, -C - resumes if the download drops
curl -L -C - -o INCLUDE.zip \
  "https://zenodo.org/records/6674324/files/INCLUDE.zip?download=1"

unzip -q INCLUDE.zip -d INCLUDE                              # -> INCLUDE/INCLUDE/*.zip
unzip -q INCLUDE/INCLUDE/Pose_Signs.zip -d INCLUDE/Pose_Signs
unzip -q INCLUDE/INCLUDE/Train_Test_Split.zip \
      -d INCLUDE/INCLUDE/Train_Test_Split                    # the official split
cd ..

find data/INCLUDE/Pose_Signs/Pose_Signs -name '*.pkl' | wc -l   # expect 4284
```

### Windows (PowerShell)

Use `curl.exe`, not `curl` — bare `curl` is a PowerShell alias for
`Invoke-WebRequest`, which takes different flags and buffers the whole 663 MB in
memory. `tar` unpacks zips far faster than `Expand-Archive` here, and both ship
with Windows 10 and 11:

```powershell
mkdir data; cd data

curl.exe -L -C - -o INCLUDE.zip `
  "https://zenodo.org/records/6674324/files/INCLUDE.zip?download=1"

mkdir INCLUDE; tar -xf INCLUDE.zip -C INCLUDE
mkdir INCLUDE\Pose_Signs; tar -xf INCLUDE\INCLUDE\Pose_Signs.zip -C INCLUDE\Pose_Signs
mkdir INCLUDE\INCLUDE\Train_Test_Split
tar -xf INCLUDE\INCLUDE\Train_Test_Split.zip -C INCLUDE\INCLUDE\Train_Test_Split
cd ..

(Get-ChildItem data\INCLUDE\Pose_Signs\Pose_Signs -Recurse -Filter *.pkl).Count  # expect 4284
```

On an older Windows without `tar`, swap in
`Expand-Archive -Path INCLUDE.zip -DestinationPath INCLUDE` (slower, same result).

### Then

The expected layout is `data/INCLUDE/Pose_Signs/Pose_Signs/<Category>/<Sign>/*.pkl`
— 4284 `.pkl` files across 263 folders, one of which is empty, leaving 262 usable
classes. Point somewhere else with `BRIDGE_DATA_ROOT`:

```bash
export BRIDGE_DATA_ROOT=/path/to/Pose_Signs        # macOS / Linux
$env:BRIDGE_DATA_ROOT = "D:\datasets\Pose_Signs"   # Windows PowerShell
```

Confirm the features look right before spending hours on a training run:

```bash
uv run python main.py diagnose
```

### The official train/test split

`INCLUDE.zip` ships `Train_Test_Split.zip`, and **it is used by default**.

It matters because INCLUDE's split is *signer-disjoint*: the people in the test
half never appear in training. A random split lets the same signer land on both
sides, so the model gets credit for recognising people rather than signs, and
validation accuracy comes out higher than the model deserves. Only the official
split is comparable to published INCLUDE results.

`train` finds the tables automatically and prints how many clips matched:

```
Matched 4284/4284 clips to the official split (3475 train, 809 test)
Using INCLUDE's official signer-disjoint split (3475 train / 809 val)
```

If the tables are missing or don't line up, it says so and falls back to the
random split rather than half-using them:

```bash
uv run python main.py train --split official   # require it; fail if unusable
uv run python main.py train --split random     # force the old random split
uv run python main.py train --split auto       # default: official if available
```

Expect **lower** validation accuracy than before. That's the point — the old
number was inflated.

## Layout

| Path | What it does |
| --- | --- |
| `main.py` | CLI — `train`, `run`, `eval`, `diagnose` |
| `src/config.py` | Paths, hyperparameters, feature layout (env-overridable) |
| `src/preprocess.py` | **The feature pipeline — used by training *and* live inference** |
| `src/dataset.py` | INCLUDE `.pkl` → `X`/`y` arrays, split, DataLoaders |
| `src/model.py` | `SignTransformer`, bundle save/load |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Metrics + plots into `results/` |
| `src/landmarks.py` | Live MediaPipe landmark extraction (75 × 3 + confidences) |
| `src/infer.py` | Segmented live recognition, video inference |
| `src/diagnose.py` | Compares training features against live camera features |

## How live inference matches training

INCLUDE Pose_Signs stores 75 landmarks per frame in MediaPipe Holistic order —
33 pose + 21 left hand + 21 right hand, `(x, y, z)` each = **225 features**.
`src/landmarks.py` rebuilds exactly that layout from the MediaPipe pose and hand
landmarker tasks.

Matching the layout isn't enough on its own, though — the *values* have to mean
the same thing too. Both paths therefore call the same `preprocess.encode()`,
which is the only place features are built:

| Step | Why |
| --- | --- |
| Frame size resolved from the data | The `.pkl` `vid_shape` tuple doesn't record which entry is width; reading it backwards distorts every skeleton by the square of the aspect ratio |
| Low-confidence landmarks dropped, then interpolated | MediaPipe emits all 33 pose points even when it can only guess them — the `confidences` array says which |
| Coordinates made body-relative | Centred on the shoulder midpoint, scaled by shoulder width, so distance from the camera and position in frame stop mattering |
| Both axes divided by frame *height* | Preserves aspect ratio, so a 16:9 clip and a 4:3 webcam agree on what a shape is |
| Whole clip resampled to 40 frames | Framerate stops mattering — a 25 fps recording and a 10 fps webcam produce the same features |

`z` is deliberately left alone: the `.pkl` files store x/y in pixels but z already
in MediaPipe's own units, which is exactly what the live landmarker returns.

Bundles record the preprocessing version they were trained under, and loading
one built by a different version fails loudly rather than producing plausible
nonsense. The cached `X.npy` re-encodes itself the same way.

> MediaPipe is pinned to `<1.0` — the 1.x vision tasks crash on macOS
> (`Check failed: service_ Service is unavailable`).

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
