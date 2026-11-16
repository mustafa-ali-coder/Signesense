"""
static_sign.py -- trainable static sign (fingerspelling) classifier.

k-nearest-neighbour over normalized landmark features. Chosen deliberately:
  * works with tiny per-class sample counts (5-15 samples/letter),
  * no training step, add/remove classes instantly,
  * distance gives a natural confidence + "unknown" rejection.

Pure Python. Persistence is plain JSON so the dataset is inspectable.
"""

import json
import os
from collections import Counter

from sign_core import feature_distance, hand_count_of, COMBINED_DIM

try:
    import numpy as _np
except ImportError:          # tests and core logic don't need numpy
    _np = None


class StaticSignClassifier:

    def __init__(self, k=3, unknown_threshold=2.0):
        """
        k: neighbours to vote.
        unknown_threshold: if the mean distance of the winning class's
            neighbours exceeds this, report None (unknown pose). 2.0 chosen
            by held-out sweep on the bundled ASL dataset.
        """
        self.k = k
        self.unknown_threshold = unknown_threshold
        self.samples = []   # list of feature vectors
        self.labels = []    # parallel list of str labels
        self._matrix = None  # numpy cache, rebuilt lazily

    # -- dataset management --------------------------------------------------

    def add_sample(self, features, label):
        if len(features) != COMBINED_DIM:
            raise ValueError(
                f"expected {COMBINED_DIM}-dim vector, got {len(features)}")
        self.samples.append(list(features))
        self.labels.append(str(label))
        self._matrix = None

    def remove_label(self, label):
        """Delete all samples of a label. Returns count removed."""
        keep_s, keep_l, removed = [], [], 0
        for s, l in zip(self.samples, self.labels):
            if l == label:
                removed += 1
            else:
                keep_s.append(s)
                keep_l.append(l)
        self.samples, self.labels = keep_s, keep_l
        self._matrix = None
        return removed

    def label_counts(self):
        return dict(Counter(self.labels))

    # -- classification ------------------------------------------------------

    def _neighbours(self, features):
        """Top-k (distance, label) with matching hand count.
        Vectorized with numpy when available; pure Python otherwise."""
        query_hands = hand_count_of(features)

        if _np is not None:
            if self._matrix is None:
                self._matrix = _np.asarray(self.samples, dtype=_np.float64)
                self._hand_counts = _np.rint(
                    self._matrix[:, -2] + self._matrix[:, -1]).astype(int)
            mask = self._hand_counts == query_hands
            if not mask.any():
                return []
            diffs = self._matrix[mask] - _np.asarray(features)
            dists = _np.sqrt((diffs * diffs).sum(axis=1))
            idx_all = _np.flatnonzero(mask)
            order = _np.argsort(dists)[:self.k]
            return [(float(dists[i]), self.labels[int(idx_all[i])])
                    for i in order]

        dists = []
        for s, l in zip(self.samples, self.labels):
            if hand_count_of(s) != query_hands:
                continue
            dists.append((feature_distance(features, s), l))
        dists.sort(key=lambda t: t[0])
        return dists[:self.k]

    def classify(self, features):
        """Returns (label, confidence) or (None, 0.0) if unknown/empty.

        Only samples with the same hand count compete: a one-handed pose
        never matches a two-handed sign.
        """
        if not self.samples:
            return None, 0.0

        top = self._neighbours(features)
        if not top:
            return None, 0.0

        votes = Counter(l for _, l in top)
        winner, _ = votes.most_common(1)[0]

        winner_dists = [d for d, l in top if l == winner]
        mean_d = sum(winner_dists) / len(winner_dists)

        if mean_d > self.unknown_threshold:
            return None, 0.0

        confidence = max(0.0, 1.0 - mean_d / self.unknown_threshold)
        return winner, confidence

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        data = {
            "format": "signsense-static-v1",
            "k": self.k,
            "unknown_threshold": self.unknown_threshold,
            "samples": self.samples,
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
        if data.get("format") != "signsense-static-v1":
            raise ValueError(f"unrecognized file format in {path}")
        obj = cls(k=data["k"], unknown_threshold=data["unknown_threshold"])
        obj.samples = data["samples"]
        obj.labels = data["labels"]
        if len(obj.samples) != len(obj.labels):
            raise ValueError("corrupt dataset: samples/labels length mismatch")
        return obj
