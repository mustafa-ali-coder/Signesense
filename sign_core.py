"""
sign_core.py -- SignSense core geometry.

Pure Python, zero dependencies. Everything here is testable without a webcam.

Landmark convention: MediaPipe Hands, 21 landmarks per hand, each (x, y, z).
Index 0 = wrist, 9 = middle finger MCP. I use x, y only (z is noisy).

Normalization pipeline (makes features invariant to where/how big/how tilted
the hand is in frame):
    1. Translate so wrist is at origin.
    2. Scale so |wrist -> middle_mcp| == 1.
    3. Rotate so wrist -> middle_mcp points "up" (negative y, image coords).

Result: 20 landmarks x 2 coords = 40-dim feature vector per hand
(wrist dropped -- it's always the origin).
"""

import math

WRIST = 0
MIDDLE_MCP = 9
NUM_LANDMARKS = 21
FEATURES_PER_HAND = (NUM_LANDMARKS - 1) * 2  # 40


class HandGeometryError(ValueError):
    """Raised when landmarks are degenerate (e.g. zero-size hand)."""


def normalize_hand(landmarks):
    """Normalize one hand's landmarks.

    landmarks: sequence of 21 (x, y) or (x, y, z) tuples.
    Returns: flat list of 40 floats (20 landmarks x 2, wrist excluded).
    Raises HandGeometryError on degenerate input.
    """
    if len(landmarks) != NUM_LANDMARKS:
        raise HandGeometryError(
            f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")

    wx, wy = landmarks[WRIST][0], landmarks[WRIST][1]
    mx, my = landmarks[MIDDLE_MCP][0], landmarks[MIDDLE_MCP][1]

    ref_dx, ref_dy = mx - wx, my - wy
    scale = math.hypot(ref_dx, ref_dy)
    if scale < 1e-9:
        raise HandGeometryError("degenerate hand: wrist == middle_mcp")

    # Rotation that maps the reference vector onto (0, -1) ("up").
    # Current angle of reference vector:
    theta = math.atan2(ref_dy, ref_dx)
    # Target angle is -pi/2. Rotate by (-pi/2 - theta).
    rot = -math.pi / 2.0 - theta
    cos_r, sin_r = math.cos(rot), math.sin(rot)

    out = []
    for i in range(NUM_LANDMARKS):
        if i == WRIST:
            continue
        x = (landmarks[i][0] - wx) / scale
        y = (landmarks[i][1] - wy) / scale
        # rotate
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        out.append(rx)
        out.append(ry)
    return out


def feature_distance(a, b):
    """Euclidean distance between two feature vectors of equal length."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Two-hand handling.
#
# Many PSL signs are two-handed. I build a fixed-width combined vector:
#   [left shape 40][left orient 2][right shape 40][right orient 2][flags 2]
# Missing hand -> zeros for its blocks, flag 0.0.
#
# Orientation dims: shape features are deliberately rotation-invariant, but
# some fingerspelling letters differ ONLY by which way the hand points
# (ASL G vs Q, H vs U, K vs P). So the global angle of wrist->middle_mcp is
# appended as (cos, sin) * ORIENT_WEIGHT: small tilts cost little, pointing
# a different direction costs a lot.
# ---------------------------------------------------------------------------

ORIENT_WEIGHT = 0.3
HAND_BLOCK = FEATURES_PER_HAND + 2            # 42
COMBINED_DIM = HAND_BLOCK * 2 + 2             # 86


def hand_orientation(landmarks):
    """(cos, sin) of the wrist->middle_mcp angle in image coordinates."""
    dx = landmarks[MIDDLE_MCP][0] - landmarks[WRIST][0]
    dy = landmarks[MIDDLE_MCP][1] - landmarks[WRIST][1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        raise HandGeometryError("degenerate hand: wrist == middle_mcp")
    return dx / n, dy / n


def combine_hands(left=None, right=None):
    """Build the fixed-width combined feature vector.

    left / right: 21-landmark sequences or None.
    Returns: list of 86 floats.
    """
    if left is None and right is None:
        raise HandGeometryError("no hands provided")

    vec = []
    for hand in (left, right):
        if hand is None:
            vec.extend([0.0] * HAND_BLOCK)
        else:
            vec.extend(normalize_hand(hand))
            c, s = hand_orientation(hand)
            vec.append(c * ORIENT_WEIGHT)
            vec.append(s * ORIENT_WEIGHT)
    vec.append(1.0 if left is not None else 0.0)
    vec.append(1.0 if right is not None else 0.0)
    return vec


def hand_count_of(vec):
    """Number of hands encoded in a combined vector (reads the flags)."""
    return int(round(vec[-2])) + int(round(vec[-1]))


# ---------------------------------------------------------------------------
# Trajectory normalization (for dynamic signs).
#
# Per-frame shape normalization deliberately throws away *where* the hand is.
# But dynamic signs often differ only by hand path (circle vs. swipe). So a
# dynamic sample also carries the wrist trajectory, normalized to its own
# bounding box, preserving path shape while staying position/scale invariant.
# ---------------------------------------------------------------------------

def normalize_trajectory(points, min_extent=1e-6):
    """Normalize a wrist trajectory to fit centered in [-0.5, 0.5]^2.

    points: sequence of (x, y).
    Uniform scale (aspect ratio preserved). Near-stationary trajectories
    collapse to all-zeros -- that's correct: 'no movement' is the shape.
    """
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    extent = max(max_x - min_x, max_y - min_y)
    if extent < min_extent:
        return [(0.0, 0.0) for _ in points]
    return [((p[0] - cx) / extent, (p[1] - cy) / extent) for p in points]


def resample_sequence(seq, target_len):
    """Linearly resample a sequence of equal-length float vectors to
    target_len frames. Handles len(seq) == 1 by repetition."""
    if not seq:
        raise ValueError("empty sequence")
    if target_len < 1:
        raise ValueError("target_len must be >= 1")
    n = len(seq)
    if n == 1:
        return [list(seq[0]) for _ in range(target_len)]
    out = []
    for i in range(target_len):
        # position in source coordinates
        t = i * (n - 1) / (target_len - 1) if target_len > 1 else 0.0
        lo = int(math.floor(t))
        hi = min(lo + 1, n - 1)
        frac = t - lo
        frame = [a + (b - a) * frac for a, b in zip(seq[lo], seq[hi])]
        out.append(frame)
    return out
