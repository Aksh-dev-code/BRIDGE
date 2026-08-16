"""Compare training features against live webcam features.

The model can only work if the numbers it sees live look like the numbers it
trained on. This walks a sample of the .pkl dataset, works out how the raw
keypoints must be normalized, then captures a few seconds from the camera and
prints the two distributions side by side.

    python main.py diagnose

Run it on the machine that has the INCLUDE dataset.
"""

import os
import pickle
import random

import numpy as np

from src import config, preprocess

# MediaPipe pose indices, used to check that a skeleton has human proportions.
L_SHOULDER, R_SHOULDER, L_HIP = 11, 12, 23


def _sample_pkls(data_root, n=40, seed=0):
    found = []
    for dirpath, _, filenames in os.walk(data_root):
        found.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".pkl"))
    random.Random(seed).shuffle(found)
    return found[:n], len(found)


def _orientation_vote(kp, conf, vid_shape):
    """Score both readings of vid_shape by how many confident points land in frame."""
    a, b = float(vid_shape[0]), float(vid_shape[1])
    scores = preprocess.score_frame_sizes(kp, conf, vid_shape)
    if any(np.isnan(v) for v in scores.values()):
        return None
    return {"(h,w) — current code": scores[(b, a)], "(w,h) — swapped": scores[(a, b)]}


def inspect_dataset(n=40):
    paths, total = _sample_pkls(config.DATA_ROOT, n)
    if not paths:
        raise FileNotFoundError(
            f"No .pkl files under {config.DATA_ROOT}.\n"
            "Point BRIDGE_DATA_ROOT at your INCLUDE copy and re-run."
        )
    print(f"Found {total} .pkl files — inspecting {len(paths)}\n")

    votes = {"(h,w) — current code": [], "(w,h) — swapped": []}
    shapes, conf_all = set(), []

    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        kp = np.asarray(d["keypoints"], dtype=np.float64)
        conf = np.asarray(d.get("confidences", np.ones(kp.shape[:2])), dtype=np.float64)
        vs = tuple(d["vid_shape"][:2])
        shapes.add(vs)
        conf_all.append(conf.ravel())

        v = _orientation_vote(kp, conf, vs)
        if v:
            for k, s in v.items():
                votes[k].append(s)

    conf_all = np.concatenate(conf_all)
    print("--- vid_shape ---")
    print(f"  distinct values across sample: {sorted(shapes)[:5]}"
          f"{' ...' if len(shapes) > 5 else ''}")

    print("\n--- which reading of vid_shape is right? ---")
    print("  (fraction of confidently-detected landmarks that land inside the frame)")
    means = {k: float(np.mean(v)) for k, v in votes.items() if v}
    for k, m in means.items():
        print(f"    {k:24s}  {m:6.1%}")
    if means:
        cur = means.get("(h,w) — current code", 0.0)
        swp = means.get("(w,h) — swapped", 0.0)
        if abs(swp - cur) <= 0.02:
            print("\n  => INCONCLUSIVE: both readings fit. That happens when the signer")
            print("     sits small and centred, so no landmark falls outside either way.")
            print("     preprocess.py keeps the (h,w) reading in this case.")
        elif swp > cur:
            print("\n  => vid_shape is (width, height) — the opposite of the old code.")
            print("     preprocess.resolve_frame_size detects this per clip, so it is")
            print("     already handled; this is just confirmation.")
        else:
            print("\n  => vid_shape is (height, width), as the notebook assumed.")

    print("\n--- confidences ---")
    print(f"  mean {conf_all.mean():.3f} | median {np.median(conf_all):.3f}")
    print(f"  landmarks below {preprocess.CONF_THRESHOLD} confidence: "
          f"{(conf_all < preprocess.CONF_THRESHOLD).mean():.1%}")
    print(f"  landmarks at exactly 0.0:      {(conf_all == 0).mean():.1%}")
    print("  ^ points MediaPipe guessed at; preprocess.encode drops and interpolates these")

    return {"shapes": shapes, "votes": votes, "conf": conf_all}


def training_feature_stats(n=40):
    """Value distribution of X exactly as dataset.py builds it today."""
    from src.dataset import load_pkl_sequence

    paths, _ = _sample_pkls(config.DATA_ROOT, n)
    encoded = [load_pkl_sequence(p) for p in paths]
    usable = [s for s in encoded if s is not None]
    if not usable:
        raise RuntimeError(
            f"None of the {len(paths)} sampled clips could be encoded — no frame in "
            "any of them had both shoulders detected."
        )
    if len(usable) < len(encoded):
        print(f"  ({len(encoded) - len(usable)} of {len(encoded)} sampled clips "
              "couldn't be encoded — no shoulders detected)")

    seqs = np.stack(usable)
    nz = seqs[seqs != 0]
    return {
        "min": float(seqs.min()), "max": float(seqs.max()),
        "mean": float(nz.mean()), "std": float(nz.std()),
        "p1": float(np.percentile(nz, 1)), "p99": float(np.percentile(nz, 99)),
        "zero_frac": float((seqs == 0).mean()),
    }


def live_feature_stats(seconds=6.0, source=0, swap_hands=False):
    """Same statistics, from the webcam, through the same preprocess.encode call.

    Both sides have to go through the identical encoder or the comparison means
    nothing — raw landmarks and encoded features live on completely different
    scales, so they would always look mismatched.
    """
    import time

    import cv2

    from src.landmarks import LandmarkExtractor

    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source '{source}'.")

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect = float(w / h) if h else 1.0
    print(f"\nCapturing {seconds:.0f}s from the camera ({int(w)}x{int(h)}, "
          f"aspect {aspect:.2f}) — sign something.")

    kps, confs, hands, t0 = [], [], 0, time.monotonic()
    with LandmarkExtractor(swap_hands=swap_hands) as ex:
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            kp, conf, hands_seen = ex.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                              int((time.monotonic() - t0) * 1000))
            kps.append(kp)
            confs.append(conf)
            hands += bool(hands_seen)
    cap.release()

    if not kps:
        raise RuntimeError("No frames captured.")
    if hands == 0:
        print("  (no hands were detected in that capture — the numbers below are "
              "pose-only)")

    seqs = preprocess.encode(np.stack(kps), np.stack(confs), aspect=aspect)
    if seqs is None:
        raise RuntimeError(
            "Couldn't encode the capture — no frame had both shoulders visible. "
            "Sit so your torso is in frame and try again."
        )

    nz = seqs[seqs != 0]
    return {
        "min": float(seqs.min()), "max": float(seqs.max()),
        "mean": float(nz.mean()), "std": float(nz.std()),
        "p1": float(np.percentile(nz, 1)), "p99": float(np.percentile(nz, 99)),
        "zero_frac": float((seqs == 0).mean()),
        "aspect": aspect,
        "frames": len(kps), "hand_frames": hands,
    }


def diagnose(n=40, seconds=6.0, source=0, skip_live=False):
    print("=" * 64)
    print("BRIDGE — training vs live feature check")
    print("=" * 64)
    inspect_dataset(n)

    train = training_feature_stats(n)
    print("\n--- feature values as the model actually receives them ---")
    if skip_live:
        live = None
        print(f"  {'':10s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s} {'zeros':>8s}")
        print(f"  {'TRAIN':10s} {train['min']:8.3f} {train['max']:8.3f} "
              f"{train['mean']:8.3f} {train['std']:8.3f} {train['zero_frac']:8.1%}")
    else:
        live = live_feature_stats(seconds, source)
        print(f"\n  {'':10s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s} {'zeros':>8s}")
        for label, s in (("TRAIN", train), ("LIVE", live)):
            print(f"  {label:10s} {s['min']:8.3f} {s['max']:8.3f} "
                  f"{s['mean']:8.3f} {s['std']:8.3f} {s['zero_frac']:8.1%}")

        drift = abs(train["mean"] - live["mean"]) / max(train["std"], 1e-6)
        print(f"\n  mean shift between train and live: {drift:.2f} standard deviations")
        if drift > 0.5 or not (0.5 < live["max"] / max(train["max"], 1e-6) < 2.0):
            print("  => TRAIN AND LIVE DO NOT MATCH. The model is being fed values it")
            print("     never saw, which is why it collapses onto a few default classes.")
            print("     Fix preprocessing on BOTH sides, then retrain.")
        else:
            print("  => distributions broadly agree; the gap is elsewhere.")

    print("=" * 64)
    return {"train": train, "live": live}
