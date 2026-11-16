"""
dynamic_sign.py -- trainable dynamic (motion) sign classifier.

Dynamic Time Warping nearest-neighbour over sequences of per-frame features.
Why DTW instead of an LSTM/transformer: you have ~5 samples per sign, not
5000. DTW is exact, needs no training, and tolerates speed variation --
signing the same word faster or slower still matches.

Per-frame feature = shape features (normalized hand pose, 82 dims)
                  + wrist trajectory point (2 dims, trajectory-normalized)
                    weighted up, because path is often what distinguishes
                    dynamic signs.

Pure Python, stdlib only.
"""

import json
import math
import os
from collections import Counter

from sign_core import (combine_hands, normalize_trajectory,
                       resample_sequence, COMBINED_DIM)

try:
    import numpy as _np
except ImportError:
    _np = None

SEQ_LEN = 24          # frames after resampling
TRAJ_WEIGHT = 3.0     # multiplier on trajectory coords in frame distance
DTW_BAND = 6          # Sakoe-Chiba band radius (frames)

FRAME_DIM = COMBINED_DIM + 2  # 84


def build_sequence_features(frames):
    """Turn a raw recording into a fixed-length feature sequence.

    frames: list of dicts, each {"left": lm|None, "right": lm|None,
                                 "wrist": (x, y)}
        lm = 21-landmark list; "wrist" = raw wrist position of the primary
        hand in image coords (used for trajectory).
    Returns: list of SEQ_LEN frames, each FRAME_DIM floats.
    """
    if not frames:
        raise ValueError("empty recording")

    shape_seq = []
    traj_pts = []
    for fr in frames:
        shape_seq.append(combine_hands(fr.get("left"), fr.get("right")))
        traj_pts.append(fr["wrist"])

    traj_norm = normalize_trajectory(traj_pts)

    merged = []
    for shape, (tx, ty) in zip(shape_seq, traj_norm):
        merged.append(list(shape) + [tx * TRAJ_WEIGHT, ty * TRAJ_WEIGHT])

    return resample_sequence(merged, SEQ_LEN)


def _frame_dist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def dtw_distance(seq_a, seq_b, band=DTW_BAND):
    """DTW with Sakoe-Chiba band. Returns path-length-normalized distance.
    Uses numpy for the frame-distance matrix when available (identical
    result, much faster); falls back to pure Python."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        raise ValueError("empty sequence")

    INF = float("inf")

    if _np is not None:
        A = _np.asarray(seq_a, dtype=_np.float64)
        B = _np.asarray(seq_b, dtype=_np.float64)
        D = _np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        frame = lambda i, j: float(D[i, j])
    else:
        frame = lambda i, j: _frame_dist(seq_a[i], seq_b[j])

    # dp[i][j] = cost of best alignment of a[:i+1], b[:j+1]
    dp = [[INF] * m for _ in range(n)]
    for i in range(n):
        j_lo = max(0, i - band)
        j_hi = min(m - 1, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = frame(i, j)
            if i == 0 and j == 0:
                dp[i][j] = cost
                continue
            best = INF
            if i > 0 and dp[i - 1][j] < best:
                best = dp[i - 1][j]
            if j > 0 and dp[i][j - 1] < best:
                best = dp[i][j - 1]
            if i > 0 and j > 0 and dp[i - 1][j - 1] < best:
                best = dp[i - 1][j - 1]
            if best is INF:
                continue  # unreachable inside band edge
            dp[i][j] = cost + best

    if dp[n - 1][m - 1] == INF:
        # band too narrow for this length pair; fall back to full DTW
        return dtw_distance(seq_a, seq_b, band=max(n, m))
    return dp[n - 1][m - 1] / (n + m)


class DynamicSignClassifier:

    def __init__(self, unknown_threshold=1.1):
        self.unknown_threshold = unknown_threshold
        self.sequences = []  # list of SEQ_LEN x FRAME_DIM
        self.labels = []

    def add_sample(self, sequence, label):
        if len(sequence) != SEQ_LEN:
            raise ValueError(f"expected {SEQ_LEN}-frame sequence")
        for fr in sequence:
            if len(fr) != FRAME_DIM:
                raise ValueError(f"expected {FRAME_DIM}-dim frames")
        self.sequences.append([list(fr) for fr in sequence])
        self.labels.append(str(label))

    def remove_label(self, label):
        keep_s, keep_l, removed = [], [], 0
        for s, l in zip(self.sequences, self.labels):
            if l == label:
                removed += 1
            else:
                keep_s.append(s)
                keep_l.append(l)
        self.sequences, self.labels = keep_s, keep_l
        return removed

    def label_counts(self):
        return dict(Counter(self.labels))

    def classify(self, sequence):
        """Returns (label, confidence) or (None, 0.0)."""
        if not self.sequences:
            return None, 0.0
        best_d, best_l = float("inf"), None
        for s, l in zip(self.sequences, self.labels):
            d = dtw_distance(sequence, s)
            if d < best_d:
                best_d, best_l = d, l
        if best_d > self.unknown_threshold:
            return None, 0.0
        confidence = max(0.0, 1.0 - best_d / self.unknown_threshold)
        return best_l, confidence

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        data = {
            "format": "signsense-dynamic-v1",
            "unknown_threshold": self.unknown_threshold,
            "seq_len": SEQ_LEN,
            "sequences": self.sequences,
            "labels": self.labels,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != "signsense-dynamic-v1":
            raise ValueError(f"unrecognized file format in {path}")
        obj = cls(unknown_threshold=data["unknown_threshold"])
        obj.sequences = data["sequences"]
        obj.labels = data["labels"]
        if len(obj.sequences) != len(obj.labels):
            raise ValueError("corrupt dataset: length mismatch")
        return obj
