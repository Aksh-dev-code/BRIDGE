"""Inference: predict signs from a live camera or a video file."""

from collections import deque

import numpy as np
import torch

from src import config
from src.model import load_bundle


def predict_sequence(model, class_names, seq, dev, topk=3):
    """seq: (max_frames, 225) -> list of (class_name, probability), best first."""
    x = torch.tensor(np.asarray(seq, dtype=np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    k = min(topk, probs.numel())
    top = probs.topk(k)
    return [(class_names[i], float(p)) for p, i in zip(top.values.cpu(), top.indices.cpu())]


def predict_video(video_path, topk=3, swap_hands=False):
    from src.landmarks import sequence_from_video

    dev = config.device()
    model, class_names, cfg = load_bundle(map_location=dev)
    seq = sequence_from_video(video_path, max_frames=cfg["max_len"], swap_hands=swap_hands)
    preds = predict_sequence(model, class_names, seq, dev, topk=topk)

    print(f"\nPredictions for {video_path}:")
    for name, prob in preds:
        print(f"  {name:30s} {prob:.3f}")
    return preds


def run_realtime(source=0, topk=3, threshold=0.35, swap_hands=False, mirror=True,
                 idle_reset=8, stability=5, hold=45):
    """Live recognition with gesture segmentation.

    Frames stream into a fixed-length window, but a prediction is only
    *committed* once the same label has won `stability` windows in a row above
    `threshold`. Two things end a sign and clear the window, so one sign can
    never bleed into the next:

      - dropping your hands for `idle_reset` frames (a natural boundary), and
      - committing a sign (the window restarts empty for the following one).

    Keys: Q / ESC quit, R reset the window, C clear the history.
    """
    import cv2

    from src.landmarks import LandmarkExtractor

    dev = config.device()
    model, class_names, cfg = load_bundle(map_location=dev)
    max_len = cfg["max_len"]
    stability = max(1, stability)
    print(f"Loaded model: {len(class_names)} classes, window={max_len} frames, device={dev}")

    # Allow "0"/"1" to mean a local webcam index, anything else is a URL/path.
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera source '{source}'.\n"
            "For a phone stream, check the app is running, both devices are on the\n"
            "same WiFi, and the URL matches exactly (including '/video')."
        )

    buffer = deque(maxlen=max_len)      # the last max_len frames of landmarks
    votes = deque(maxlen=stability)     # recent top-1 labels, for agreement
    history = []                        # signs committed so far
    preds = []                          # live top-k, for display only
    committed = None                    # (name, prob) of the last committed sign
    committed_at = -10_000
    idle = 0                            # consecutive frames with no hands
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("Running. Q or ESC to quit, R to reset the window, C to clear history.")
    with LandmarkExtractor(swap_hands=swap_hands) as ex:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Lost the camera feed.")
                break

            # Detect on the true (unmirrored) frame so handedness labels stay correct.
            kp, hands_seen = ex.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                        int(frame_idx * 1000 / fps))
            frame_idx += 1
            idle = 0 if hands_seen else idle + 1

            if idle >= idle_reset:
                # Hands are down: the sign is over. Start the next one clean.
                buffer.clear()
                votes.clear()
                preds = []
            else:
                buffer.append(kp.reshape(-1))

            if len(buffer) == max_len:
                preds = predict_sequence(model, class_names, np.array(buffer), dev, topk=topk)
                name, prob = preds[0]
                votes.append(name if prob >= threshold else None)

                # Commit only once the window has agreed with itself for a while;
                # mid-transition windows flicker and never reach agreement.
                if len(votes) == stability and votes[0] is not None and len(set(votes)) == 1:
                    committed = (name, prob)
                    committed_at = frame_idx
                    history.append(name)
                    print(f"  → {name}  ({prob:.2f})")
                    buffer.clear()   # don't re-read the sign we just emitted
                    votes.clear()
                    preds = []

            display = cv2.flip(frame, 1) if mirror else frame.copy()
            _draw_overlay(cv2, display, {
                "preds": preds,
                "filled": len(buffer),
                "max_len": max_len,
                "hands_seen": hands_seen,
                "committed": committed,
                "fresh": frame_idx - committed_at < hold,
                "history": history,
                "state": _state_label(idle, idle_reset, len(buffer), max_len, votes, stability),
            })
            cv2.imshow("BRIDGE — Sign Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                buffer.clear()
                votes.clear()
                preds = []
            if key in (ord("c"), ord("C")):
                history.clear()
                committed = None

    cap.release()
    cv2.destroyAllWindows()

    if history:
        print(f"\nRecognized {len(history)} signs: {' · '.join(history)}")


def _state_label(idle, idle_reset, filled, max_len, votes, stability):
    if idle >= idle_reset:
        return "IDLE — raise your hands to sign"
    if filled < max_len:
        return f"READING {filled}/{max_len}"
    agree = sum(1 for v in votes if v is not None and v == votes[-1])
    return f"CONFIRMING {agree}/{stability}"


def _draw_overlay(cv2, img, s):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    GREEN, AMBER, GREY, RED, DIM = (0, 255, 0), (0, 200, 255), (180, 180, 180), (0, 0, 255), (110, 110, 110)

    cv2.rectangle(img, (0, 0), (w, 132), (0, 0, 0), -1)

    # Last committed sign — the actual result, bright while fresh then dimmed.
    if s["committed"]:
        name, prob = s["committed"]
        cv2.putText(img, f"{name}  ({prob:.2f})", (15, 45), font, 1.0,
                    GREEN if s["fresh"] else DIM, 2)
    else:
        cv2.putText(img, "waiting for a sign...", (15, 45), font, 0.85, GREY, 2)

    # What the window is leaning toward right now (not committed yet).
    if s["preds"]:
        live = "  |  ".join(f"{n} {p:.2f}" for n, p in s["preds"][:3])
        cv2.putText(img, live, (15, 75), font, 0.55, GREY, 1)

    cv2.putText(img, s["state"], (15, 100), font, 0.55, AMBER, 1)

    hand_color = GREEN if s["hands_seen"] else RED
    cv2.putText(img, f"hands: {s['hands_seen']}", (w - 150, 30), font, 0.6, hand_color, 2)
    cv2.putText(img, "Q=quit  R=reset  C=clear", (w - 300, h - 15), font, 0.55, AMBER, 1)

    # Window-fill progress bar
    bar = int((s["filled"] / s["max_len"]) * (w - 30))
    cv2.rectangle(img, (15, 114), (w - 15, 126), (80, 80, 80), 1)
    cv2.rectangle(img, (15, 114), (15 + bar, 126), GREEN, -1)

    # Running transcript of what's been recognized
    if s["history"]:
        tail = " · ".join(s["history"][-6:])
        cv2.rectangle(img, (0, h - 46), (w, h - 30), (0, 0, 0), -1)
        cv2.putText(img, tail, (15, h - 34), font, 0.55, GREY, 1)
