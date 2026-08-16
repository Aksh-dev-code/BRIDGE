"""The one feature pipeline, shared by training and live inference.

Everything that turns landmarks into model input lives here and nowhere else.
`dataset.py` calls it on .pkl clips, `infer.py` calls it on webcam frames, and
because it is the same code both times the two can no longer drift apart —
which is exactly how the model ended up being fed values it had never seen.

Four things happen to every clip, in this order:

1. **Frame size is resolved from the data.** The .pkl `vid_shape` tuple does not
   say which entry is width, so we try both and keep whichever puts confidently
   detected landmarks inside the frame.
2. **Low-confidence landmarks are dropped and interpolated.** MediaPipe emits
   all 33 pose landmarks whether or not it can see them; for an upper-body sign
   video it extrapolates hips and legs far off-screen. Those are guesses, and
   the `confidences` array says so.
3. **Coordinates go body-relative.** Centred on the shoulder midpoint and scaled
   by shoulder width, so where you sit and how far from the camera stop
   mattering. x and y are both divided by frame *height* first, which preserves
   aspect ratio — a 16:9 clip and a 4:3 webcam then agree on what a shape is.
4. **Time is resampled to a fixed length.** A whole sign maps onto MAX_FRAMES
   regardless of how long it took or what framerate recorded it.

z is deliberately left untouched. The .pkl files store x/y in pixels but z in
MediaPipe's own normalized units, which is exactly what the live landmarker
returns — so raw z already matches on both sides, and rescaling it would break
that.
"""

import numpy as np

from src import config

# MediaPipe pose landmark indices.
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12

CONF_THRESHOLD = 0.5
MIN_SCALE = 1e-4

# Shoulder width divided by the vertical drop from nose to shoulder line, for a
# real upper body. Measured at 1.755 (p10 1.64, p90 1.86) over 120 INCLUDE clips
# once their axes are corrected, and the same figure falls out of live MediaPipe.
#
# It exists because a dataset cannot be trusted to say how its axes were scaled.
# INCLUDE's .pkl files scale x by vid_shape[0] and y by vid_shape[1], but the
# video is 1920x1080 landscape — width and height were swapped during
# denormalization, leaving every skeleton squashed by (1920/1080)^2 = 3.16x.
# Nothing in the file records that. Rather than special-casing one dataset,
# both paths measure this ratio and rescale x until it holds, which makes any
# source self-correcting.
HUMAN_SHOULDER_TO_HEAD = 1.755
_MIN_HEAD_DROP = 1e-6


def score_frame_sizes(kp, conf, vid_shape, conf_thresh=CONF_THRESHOLD):
    """Score both readings of vid_shape -> {(w, h): fraction of points in frame}.

    Landmarks the detector was confident about must lie inside the frame, so a
    reading that pushes them outside it is the wrong one.
    """
    a, b = float(vid_shape[0]), float(vid_shape[1])
    good = conf >= conf_thresh
    scores = {}
    for w, h in ((b, a), (a, b)):
        if not good.any():
            scores[(w, h)] = float("nan")
            continue
        x = kp[..., 0][good] / w
        y = kp[..., 1][good] / h
        scores[(w, h)] = float(((x >= -0.05) & (x <= 1.05)
                                & (y >= -0.05) & (y <= 1.05)).mean())
    return scores


def resolve_frame_size(kp, conf, vid_shape, conf_thresh=CONF_THRESHOLD, tie_margin=0.02):
    """Work out (width, height) from the data instead of trusting tuple order.

    `vid_shape` is stored as a bare 2-tuple with no record of which entry is
    which, and reading it backwards distorts every skeleton by the square of the
    aspect ratio.

    The test only discriminates when the signer spans enough of the frame for
    one reading to push landmarks outside it. A subject sitting small and
    centred fits either way, and that tie is not evidence — it falls back to
    the notebook's (h, w) reading, and `diagnose` reports it as inconclusive
    rather than pretending to have decided.
    """
    a, b = float(vid_shape[0]), float(vid_shape[1])
    if a == b:
        return a, b

    default = (b, a)  # the notebook's reading: h = vid_shape[0], w = vid_shape[1]
    scores = score_frame_sizes(kp, conf, vid_shape, conf_thresh)
    if any(np.isnan(v) for v in scores.values()):
        return default

    (w1, h1), s1 = max(scores.items(), key=lambda kv: kv[1])
    s_default = scores[default]
    return (w1, h1) if s1 - s_default > tie_margin else default


def estimate_x_scale(kp, valid):
    """Factor to multiply x by so the skeleton has human proportions.

    Both axes arrive normalized to roughly [0, 1] by their own dimension, which
    squashes or stretches x by the frame's aspect ratio — and, for INCLUDE, by
    the aspect ratio a second time because the dataset swapped width and height.
    Comparing a horizontal body measurement against a vertical one recovers the
    combined factor without needing to know either cause.

    Returns None when the head or shoulders were never seen together, leaving
    the caller to fall back on the frame aspect.
    """
    usable = valid[:, NOSE] & valid[:, L_SHOULDER] & valid[:, R_SHOULDER]
    if not usable.any():
        return None

    k = kp[usable]
    shoulder_width = np.abs(k[:, L_SHOULDER, 0] - k[:, R_SHOULDER, 0])
    head_drop = np.abs((k[:, L_SHOULDER, 1] + k[:, R_SHOULDER, 1]) / 2 - k[:, NOSE, 1])

    ok = head_drop > _MIN_HEAD_DROP
    if not ok.any():
        return None

    measured = float(np.median(shoulder_width[ok] / head_drop[ok]))
    if not np.isfinite(measured) or measured <= 0:
        return None
    return HUMAN_SHOULDER_TO_HEAD / measured


def interpolate_missing(kp, valid):
    """Fill in landmarks the detector lost, per landmark, along time.

    A hand that blinks out for a few frames is better recovered from its
    neighbours than replaced with zeros, which the model would read as a real
    position at the origin.
    """
    kp = kp.copy()
    n_frames = kp.shape[0]
    frames = np.arange(n_frames)

    for lm in range(kp.shape[1]):
        seen = valid[:, lm]
        if seen.all():
            continue
        if not seen.any():
            kp[:, lm, :] = np.nan  # never detected — caller decides what to do
            continue
        for axis in range(kp.shape[2]):
            kp[:, lm, axis] = np.interp(frames, frames[seen], kp[seen, lm, axis])
    return kp


def body_normalize(kp, valid):
    """Centre on the shoulder midpoint and scale by shoulder width.

    Returns (normalized kp, ok) where ok is False when no frame had both
    shoulders — without them there is no reference length and the clip is
    unusable.
    """
    kp = kp.copy()
    shoulders_ok = valid[:, L_SHOULDER] & valid[:, R_SHOULDER]
    if not shoulders_ok.any():
        return kp, False

    centre = (kp[:, L_SHOULDER, :2] + kp[:, R_SHOULDER, :2]) / 2.0
    scale = np.linalg.norm(kp[:, L_SHOULDER, :2] - kp[:, R_SHOULDER, :2], axis=1)

    # Frames that lost a shoulder borrow the anchor from frames that kept it,
    # so a momentary pose dropout doesn't jump the whole skeleton.
    frames = np.arange(len(kp))
    for axis in range(2):
        centre[:, axis] = np.interp(frames, frames[shoulders_ok], centre[shoulders_ok, axis])
    scale = np.interp(frames, frames[shoulders_ok], scale[shoulders_ok])
    scale = np.maximum(scale, MIN_SCALE)

    kp[..., 0] = (kp[..., 0] - centre[:, None, 0]) / scale[:, None]
    kp[..., 1] = (kp[..., 1] - centre[:, None, 1]) / scale[:, None]
    return kp, True


def resample_time(seq, n_frames):
    """Uniformly resample (F, ...) onto n_frames along time.

    Training clips ran ~25 fps and webcams manage far less, so a fixed frame
    count only means the same thing once both are resampled onto it.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == n_frames:
        return seq
    if len(seq) == 1:
        return np.repeat(seq, n_frames, axis=0)

    src = np.linspace(0.0, 1.0, len(seq))
    dst = np.linspace(0.0, 1.0, n_frames)
    idx = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, len(seq) - 2)
    w = (dst - src[idx]) / np.maximum(src[idx + 1] - src[idx], 1e-9)
    w = w.reshape(-1, *([1] * (seq.ndim - 1)))
    return (seq[idx] * (1.0 - w) + seq[idx + 1] * w).astype(np.float32)


def encode(kp, conf=None, vid_shape=None, aspect=None, max_frames=None,
           conf_thresh=CONF_THRESHOLD):
    """Landmarks -> (max_frames, INPUT_DIM) float32, ready for the model.

    Exactly one of `vid_shape` (pixel coordinates, from a .pkl) or `aspect`
    (already-normalized coordinates, from the live landmarker) must be given.
    Returns None if the clip has no usable frame.
    """
    max_frames = max_frames or config.MAX_FRAMES
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1] != config.NUM_LANDMARKS:
        raise ValueError(f"expected (frames, {config.NUM_LANDMARKS}, 3), got {kp.shape}")
    if len(kp) == 0:
        return None

    if conf is None:
        conf = np.ones(kp.shape[:2])
    conf = np.asarray(conf, dtype=np.float64)

    # A landmark at exactly (0, 0) is the landmarker's "not found" sentinel, not
    # a real detection at the top-left corner.
    detected = ~((kp[..., 0] == 0) & (kp[..., 1] == 0))
    valid = (conf >= conf_thresh) & detected

    if vid_shape is not None:
        # Each axis has its own divisor; which entry belongs to which is what
        # resolve_frame_size works out.
        w, h = resolve_frame_size(kp, conf, vid_shape, conf_thresh)
        kp[..., 0] /= w
        kp[..., 1] /= h
        fallback = h / w if w else 1.0
    elif aspect is not None:
        fallback = aspect        # x was normalized by width, y by height
    else:
        raise ValueError("pass vid_shape (pixel coords) or aspect (normalized coords)")

    # Both axes are now ~[0, 1] by their own dimension, so x is off by the
    # aspect ratio — and by whatever else the source did to it. Measuring the
    # body directly recovers the whole factor; the frame aspect is only the
    # fallback for clips where the head or shoulders were never seen.
    x_scale = estimate_x_scale(kp, valid)
    kp[..., 0] *= x_scale if x_scale is not None else fallback

    kp = interpolate_missing(kp, valid)
    kp, ok = body_normalize(kp, valid)
    if not ok:
        return None

    # Landmarks never seen anywhere in the clip stay at the body centre, which
    # is the least misleading placeholder once coordinates are body-relative.
    kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)

    seq = kp.reshape(len(kp), -1).astype(np.float32)
    return resample_time(seq, max_frames)
