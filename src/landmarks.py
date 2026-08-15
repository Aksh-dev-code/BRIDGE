"""Live MediaPipe landmark extraction.

Produces the same 75-landmark layout the INCLUDE Pose_Signs .pkl files use —
33 pose + 21 left hand + 21 right hand, (x, y, z) normalized — so a webcam
frame becomes a 225-dim vector the trained transformer already understands.
"""

import atexit
import os
import re
import threading
import urllib.request

import numpy as np

from src import config

_NOISE = (
    re.compile(rb"^[IW]\d{4} "),                        # glog info / warning
    re.compile(rb"^E\d{4}.*clearcut", re.I),            # telemetry upload failures
    re.compile(rb"^=== Source Location Trace"),
    re.compile(rb"^\s*wireless/android/play/playlog"),  # trace body for the above
    re.compile(rb"^INFO: Created TensorFlow Lite"),
)

_filter_installed = False


def quiet_mediapipe_logs():
    """Filter MediaPipe's fd-2 noise while letting real errors through."""
    global _filter_installed
    if _filter_installed or os.getenv("BRIDGE_VERBOSE_MP"):
        return
    _filter_installed = True

    try:
        real_stderr = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return  # can't redirect (odd environment) — just live with the noise

    def pump():
        buf = b""
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break  # write end closed — nothing more is coming
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if not any(p.search(line) for p in _NOISE):
                    os.write(real_stderr, line + b"\n")
        if buf:
            os.write(real_stderr, buf)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()

    def shutdown():
        # Restoring fd 2 closes the pipe's only writer, so the pump sees EOF
        # and drains what's still buffered. Without the join, a final traceback
        # dies in the pipe when the daemon thread is torn down at exit.
        import sys
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(real_stderr, 2)
        except OSError:
            pass
        thread.join(timeout=2.0)

    atexit.register(shutdown)

quiet_mediapipe_logs()


def ensure_mp_models():
    """Download the MediaPipe .task files if they aren't cached yet (notebook Step 0)."""
    config.MP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((config.POSE_TASK, config.POSE_URL), (config.HAND_TASK, config.HAND_URL)):
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
    print("MediaPipe models ready:", config.POSE_TASK.exists(), config.HAND_TASK.exists())


class LandmarkExtractor:
    """Wraps the pose + hand landmarkers and emits one (75, 3) array per frame."""

    def __init__(self, swap_hands=False, min_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        ensure_mp_models()
        self.mp = mp
        self.swap_hands = swap_hands

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(config.POSE_TASK)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_confidence,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(config.HAND_TASK)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_confidence,
            )
        )

    def close(self):
        self.pose.close()
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def extract(self, frame_rgb, timestamp_ms):
        """frame_rgb: HxWx3 uint8 RGB. Returns ((75, 3) float32, hands_seen: int)."""
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)

        pose_result = self.pose.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hands.detect_for_video(mp_image, timestamp_ms)

        kp = np.zeros((config.NUM_LANDMARKS, 3), dtype=np.float32)

        if pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks[0][: config.NUM_POSE_LANDMARKS]):
                kp[i] = (lm.x, lm.y, lm.z)

        # Hand blocks start right after the pose block: left at 33, right at 54.
        left_start = config.NUM_POSE_LANDMARKS
        right_start = left_start + config.NUM_HAND_LANDMARKS

        hands_seen = 0
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name  # "Left" / "Right"
            if self.swap_hands:
                label = "Right" if label == "Left" else "Left"
            start = left_start if label == "Left" else right_start
            for i, lm in enumerate(hand_lms[: config.NUM_HAND_LANDMARKS]):
                kp[start + i] = (lm.x, lm.y, lm.z)
            hands_seen += 1

        return kp, hands_seen


def sequence_from_video(video_path, max_frames=config.MAX_FRAMES, swap_hands=False):
    """Encode a video file into one (max_frames, 225) sequence, same rules as training."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    with LandmarkExtractor(swap_hands=swap_hands) as ex:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kp, _ = ex.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(idx * 1000 / fps))
            frames.append(kp.reshape(-1))
            idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    seq = np.array(frames, dtype=np.float32)
    if len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq[:max_frames]
