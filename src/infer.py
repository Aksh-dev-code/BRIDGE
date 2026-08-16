"""Inference: predict signs from a live camera or a video file."""

import re
import time

import numpy as np
import torch

from src import config, preprocess
from src.model import load_bundle

# Dataset folders are named "13. Attack" (and one "Ex. Monsoon") — the index is
# a dataset artifact, not part of the sign, so strip it everywhere a name is
# shown to a human.
_PREFIX = re.compile(r"^\s*(?:\d+|Ex)\.\s*")


def pretty(name):
    return _PREFIX.sub("", str(name)).strip()


def predict_probs(model, seq, dev):
    """seq: (max_frames, 225) -> (num_classes,) probability vector."""
    x = torch.tensor(np.asarray(seq, dtype=np.float32)).unsqueeze(0).to(dev)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=1)[0].float().cpu().numpy()


def predict_sequence(model, class_names, seq, dev, topk=3):
    """seq: (max_frames, 225) -> list of (class_name, probability), best first."""
    probs = predict_probs(model, seq, dev)
    k = min(topk, len(probs))
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idx]


def predict_video(video_path, topk=3, swap_hands=False):
    from src.landmarks import sequence_from_video

    dev = config.device()
    model, class_names, cfg = load_bundle(map_location=dev)
    seq = sequence_from_video(video_path, max_frames=cfg["max_len"], swap_hands=swap_hands)
    preds = predict_sequence(model, class_names, seq, dev, topk=topk)

    print(f"\nPredictions for {video_path}:")
    for name, prob in preds:
        print(f"  {pretty(name):30s} {prob:.3f}")
    return preds


def run_realtime(source=0, topk=3, threshold=0.5, margin=0.15, swap_hands=False, mirror=True,
                 min_duration=0.5, max_duration=6.0, end_silence=0.5, hold=2.0, debug=False):
    """Live recognition, segmented one sign at a time.

    Training encodes a whole clip — start of sign to end — resampled onto
    MAX_FRAMES. So inference does the same thing: raise your hands to open a
    gesture, sign it, drop your hands to close it. The completed gesture is then
    passed through the identical preprocess.encode call the dataset uses.

    That is the whole reason for segmenting rather than classifying a sliding
    window: a window covering half a sign is not something the model was ever
    trained on, no matter how the confidence thresholds are tuned.

    Keys: Q / ESC quit, R drop the current gesture, C clear the history,
    D toggle debug.
    """
    import cv2

    from src.landmarks import LandmarkExtractor

    dev = config.device()
    model, class_names, cfg = load_bundle(map_location=dev)
    max_len = cfg["max_len"]
    names = [pretty(n) for n in class_names]

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera source '{source}'.\n"
            "For a phone stream, check the app is running, both devices are on the\n"
            "same WiFi, and the URL matches exactly (including '/video')."
        )

    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640.0
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480.0
    aspect = frame_w / frame_h
    print(f"Loaded model: {len(names)} classes, {max_len} frames per sign, device={dev}")
    print(f"Camera {int(frame_w)}x{int(frame_h)} (aspect {aspect:.2f})")

    kps, confs, seen = [], [], []   # the gesture currently being recorded
    started_at = None
    last_hands_at = None
    history = []
    preds = []               # preview of the in-progress gesture
    committed = None
    committed_at = -1e9
    last_preview = 0.0
    fps_ema = None
    t0 = time.monotonic()
    last_t = t0

    def classify(k, c, s):
        # Drop the trailing frames where the hands had already gone. They are
        # what ends the gesture, not part of it, and leaving them in appends a
        # frozen tail that no training clip has.
        end = len(s) - 1
        while end >= 0 and not s[end]:
            end -= 1
        if end < 0:
            return None
        seq = preprocess.encode(np.stack(k[:end + 1]), np.stack(c[:end + 1]),
                                aspect=aspect, max_frames=max_len)
        if seq is None:
            return None
        probs = predict_probs(model, seq, dev)
        order = np.argsort(probs)[::-1]
        return [(names[i], float(probs[i])) for i in order[:max(topk, 2)]]

    print("Running. Raise your hands to start a sign, lower them to finish it.")
    print("Q or ESC to quit, R to drop the current gesture, C to clear history.")
    with LandmarkExtractor(swap_hands=swap_hands) as ex:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Lost the camera feed.")
                break

            now = time.monotonic()
            dt, last_t = now - last_t, now
            if dt > 0:
                fps_ema = 1.0 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt

            # Detect on the true (unmirrored) frame so handedness stays correct.
            kp, conf, hands_seen = ex.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                              int((now - t0) * 1000))

            if hands_seen:
                if started_at is None:
                    started_at = now
                    kps, confs, seen, preds = [], [], [], []
                last_hands_at = now

            recording = started_at is not None
            if recording:
                # Keep buffering through brief detection dropouts — a hand lost
                # mid-motion is a detector hiccup, not the end of the sign.
                kps.append(kp)
                confs.append(conf)
                seen.append(bool(hands_seen))

            duration = (now - started_at) if recording else 0.0
            quiet = (now - last_hands_at) if last_hands_at is not None else 0.0
            finished = recording and (quiet >= end_silence or duration >= max_duration)

            if finished:
                result = classify(kps, confs, seen) if duration >= min_duration else None
                if result is None:
                    if duration >= min_duration:
                        print("  (couldn't read that one — keep your shoulders in frame)")
                else:
                    (n1, p1), (_, p2) = result[0], result[1]
                    if p1 >= threshold and p1 - p2 >= margin:
                        committed = (n1, p1)
                        committed_at = now
                        history.append(n1)
                        print(f"  → {n1}  ({p1:.2f})")
                    else:
                        print(f"  ? unsure: {n1} {p1:.2f} / {result[1][0]} {p2:.2f}")
                kps, confs, seen, preds = [], [], [], []
                started_at = None

            # Preview the partial gesture a few times a second, purely for the
            # display — only the completed gesture is ever committed.
            elif recording and duration >= min_duration and now - last_preview > 0.25:
                last_preview = now
                preds = classify(kps, confs, seen) or []

            display = cv2.flip(frame, 1) if mirror else frame.copy()
            _draw_overlay(cv2, display, {
                "preds": preds[:topk],
                "recording": recording,
                "duration": duration,
                "max_duration": max_duration,
                "hands_seen": hands_seen,
                "committed": committed,
                "fresh": now - committed_at < hold,
                "history": history,
                "state": _state_label(recording, duration, min_duration, quiet, end_silence),
                "debug": debug,
                "fps": fps_ema or 0.0,
                "buffered": len(kps),
            })
            cv2.imshow("BRIDGE — Sign Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                kps, confs, seen, preds, started_at = [], [], [], [], None
            if key in (ord("c"), ord("C")):
                history.clear()
                committed = None
            if key in (ord("d"), ord("D")):
                debug = not debug

    cap.release()
    cv2.destroyAllWindows()

    if history:
        print(f"\nRecognized {len(history)} signs: {' · '.join(history)}")


def _state_label(recording, duration, min_duration, quiet, end_silence):
    if not recording:
        return "READY — raise your hands to start a sign"
    if duration < min_duration:
        return "RECORDING — keep going"
    if quiet > 0.05:
        return f"FINISHING in {max(0.0, end_silence - quiet):.1f}s — hands down to commit"
    return f"RECORDING {duration:.1f}s — lower your hands when done"


def _draw_overlay(cv2, img, s):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    GREEN, AMBER, GREY, RED, DIM = (0, 255, 0), (0, 200, 255), (180, 180, 180), (0, 0, 255), (110, 110, 110)

    cv2.rectangle(img, (0, 0), (w, 132), (0, 0, 0), -1)

    if s["committed"] and s["fresh"]:
        name, prob = s["committed"]
        cv2.putText(img, f"{name}  ({prob:.2f})", (15, 45), font, 1.0, GREEN, 2)
    elif s["preds"]:
        name, prob = s["preds"][0]
        cv2.putText(img, f"{name}?  ({prob:.2f})", (15, 45), font, 0.9, AMBER, 2)
    elif s["committed"]:
        name, prob = s["committed"]
        cv2.putText(img, f"{name}  ({prob:.2f})", (15, 45), font, 1.0, DIM, 2)
    else:
        cv2.putText(img, "waiting for a sign...", (15, 45), font, 0.85, GREY, 2)

    if s["preds"][1:]:
        live = "  |  ".join(f"{n} {p:.2f}" for n, p in s["preds"][1:3])
        cv2.putText(img, live, (15, 75), font, 0.55, GREY, 1)

    cv2.putText(img, s["state"], (15, 100), font, 0.55, AMBER, 1)

    hand_color = GREEN if s["hands_seen"] else RED
    cv2.putText(img, f"hands: {s['hands_seen']}", (w - 150, 30), font, 0.6, hand_color, 2)
    if s["debug"]:
        cv2.putText(img, f"{s['fps']:.1f} fps  frames={s['buffered']}",
                    (w - 300, 55), font, 0.5, GREY, 1)
    cv2.putText(img, "Q=quit  R=drop  C=clear  D=debug", (w - 380, h - 15), font, 0.55, AMBER, 1)

    # How long the current gesture has been running, against the cap.
    frac = min(1.0, s["duration"] / s["max_duration"]) if s["max_duration"] else 0.0
    cv2.rectangle(img, (15, 114), (w - 15, 126), (80, 80, 80), 1)
    if s["recording"]:
        cv2.rectangle(img, (15, 114), (15 + int(frac * (w - 30)), 126), GREEN, -1)

    if s["history"]:
        tail = " · ".join(s["history"][-6:])
        cv2.rectangle(img, (0, h - 46), (w, h - 30), (0, 0, 0), -1)
        cv2.putText(img, tail, (15, h - 34), font, 0.55, GREY, 1)
