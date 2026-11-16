"""
test_signs.py -- SignSense test suite. No webcam, no mediapipe, no opencv.

    python test_signs.py
"""

import math
import os
import random
import tempfile
import unittest

from sign_core import (normalize_hand, combine_hands, hand_count_of,
                       normalize_trajectory, resample_sequence,
                       feature_distance, HandGeometryError,
                       COMBINED_DIM, FEATURES_PER_HAND)
from static_sign import StaticSignClassifier
from dynamic_sign import (DynamicSignClassifier, build_sequence_features,
                          dtw_distance, SEQ_LEN, FRAME_DIM)
from speller import Speller


# ---------------------------------------------------------------------------
# synthetic hand factory
# ---------------------------------------------------------------------------

def make_hand(spread=1.0, curl=0.0, seed=None):
    """Build a plausible synthetic 21-landmark hand.

    spread: finger splay multiplier. curl: 0=open, 1=fist-ish.
    Deterministic unless seed given (then small jitter is added).
    """
    rng = random.Random(seed) if seed is not None else None
    lms = [(0.5, 0.8, 0.0)]  # wrist
    finger_x = [0.38, 0.45, 0.5, 0.55, 0.62]  # thumb..pinky base x
    for f in range(5):
        base_x = 0.5 + (finger_x[f] - 0.5) * spread
        for j in range(1, 5):  # 4 joints per finger
            # fingers extend upward (decreasing y); curl folds the joints
            # ABOVE the knuckle back down. The MCP row (j==1) stays put --
            # real fists don't move the knuckles, and MCP is the scale
            # reference for normalization.
            ext = j * 0.08
            if j == 1:
                y = 0.8 - ext
            else:
                fold = curl * ((j - 1) / 3.0) * 1.5
                y = 0.8 - ext * (1.0 - fold)
            x = base_x + (0.01 * j * (f - 2) * 0.2 * spread)
            if rng:
                x += rng.uniform(-0.003, 0.003)
                y += rng.uniform(-0.003, 0.003)
            lms.append((x, y, 0.0))
    return lms


def transform_hand(lms, angle=0.0, scale=1.0, dx=0.0, dy=0.0):
    """Rotate about the wrist, scale, translate."""
    wx, wy = lms[0][0], lms[0][1]
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for x, y, z in lms:
        px, py = (x - wx) * scale, (y - wy) * scale
        rx = px * c - py * s
        ry = px * s + py * c
        out.append((rx + wx + dx, ry + wy + dy, z))
    return out


# ---------------------------------------------------------------------------
# sign_core
# ---------------------------------------------------------------------------

class TestNormalization(unittest.TestCase):

    def test_output_shape(self):
        f = normalize_hand(make_hand())
        self.assertEqual(len(f), FEATURES_PER_HAND)

    def test_translation_invariance(self):
        h = make_hand()
        f1 = normalize_hand(h)
        f2 = normalize_hand(transform_hand(h, dx=0.3, dy=-0.2))
        self.assertLess(feature_distance(f1, f2), 1e-9)

    def test_scale_invariance(self):
        h = make_hand()
        f1 = normalize_hand(h)
        f2 = normalize_hand(transform_hand(h, scale=2.7))
        self.assertLess(feature_distance(f1, f2), 1e-9)

    def test_rotation_invariance(self):
        h = make_hand()
        f1 = normalize_hand(h)
        for angle in (0.3, 1.1, -0.8, math.pi / 2):
            f2 = normalize_hand(transform_hand(h, angle=angle))
            self.assertLess(feature_distance(f1, f2), 1e-9,
                            msg=f"failed at angle={angle}")

    def test_combined_transform_invariance(self):
        h = make_hand()
        f1 = normalize_hand(h)
        f2 = normalize_hand(
            transform_hand(h, angle=0.6, scale=0.4, dx=0.1, dy=0.25))
        self.assertLess(feature_distance(f1, f2), 1e-9)

    def test_different_poses_differ(self):
        open_hand = normalize_hand(make_hand(curl=0.0))
        fist = normalize_hand(make_hand(curl=1.0))
        self.assertGreater(feature_distance(open_hand, fist), 0.5)

    def test_wrong_landmark_count_raises(self):
        with self.assertRaises(HandGeometryError):
            normalize_hand([(0, 0)] * 20)

    def test_degenerate_hand_raises(self):
        h = make_hand()
        h[9] = h[0]  # middle_mcp == wrist
        with self.assertRaises(HandGeometryError):
            normalize_hand(h)


class TestCombineHands(unittest.TestCase):

    def test_dimensions(self):
        v = combine_hands(make_hand(), make_hand())
        self.assertEqual(len(v), COMBINED_DIM)

    def test_hand_flags(self):
        both = combine_hands(make_hand(), make_hand())
        left_only = combine_hands(make_hand(), None)
        right_only = combine_hands(None, make_hand())
        self.assertEqual(hand_count_of(both), 2)
        self.assertEqual(hand_count_of(left_only), 1)
        self.assertEqual(hand_count_of(right_only), 1)
        self.assertEqual(both[-2:], [1.0, 1.0])
        self.assertEqual(left_only[-2:], [1.0, 0.0])
        self.assertEqual(right_only[-2:], [0.0, 1.0])

    def test_missing_hand_zero_block(self):
        v = combine_hands(None, make_hand())
        from sign_core import HAND_BLOCK
        self.assertTrue(all(x == 0.0 for x in v[:HAND_BLOCK]))

    def test_orientation_distinguishes_direction(self):
        """Same handshape pointing up vs down must differ by exactly the
        orientation contribution (ASL G vs Q class of confusions)."""
        from sign_core import ORIENT_WEIGHT
        h = make_hand(curl=0.5)
        up = combine_hands(None, h)
        down = combine_hands(None, transform_hand(h, angle=math.pi))
        # shape features identical; (cos,sin) flips sign -> 2 * weight
        self.assertAlmostEqual(feature_distance(up, down),
                               2 * ORIENT_WEIGHT, places=6)

    def test_small_tilt_stays_cheap(self):
        from sign_core import ORIENT_WEIGHT
        h = make_hand(curl=0.5)
        up = combine_hands(None, h)
        tilted = combine_hands(None, transform_hand(h, angle=0.12))  # ~7 deg
        self.assertLess(feature_distance(up, tilted),
                        0.5 * ORIENT_WEIGHT)

    def test_no_hands_raises(self):
        with self.assertRaises(HandGeometryError):
            combine_hands(None, None)


class TestTrajectory(unittest.TestCase):

    def test_fits_unit_box(self):
        pts = [(3.0, 5.0), (7.0, 5.0), (7.0, 9.0), (3.0, 9.0)]
        out = normalize_trajectory(pts)
        for x, y in out:
            self.assertLessEqual(abs(x), 0.5 + 1e-9)
            self.assertLessEqual(abs(y), 0.5 + 1e-9)

    def test_scale_and_translation_invariant(self):
        pts = [(0.1, 0.1), (0.2, 0.15), (0.3, 0.4)]
        big = [(x * 10 + 5, y * 10 - 3) for x, y in pts]
        a, b = normalize_trajectory(pts), normalize_trajectory(big)
        for (x1, y1), (x2, y2) in zip(a, b):
            self.assertAlmostEqual(x1, x2, places=9)
            self.assertAlmostEqual(y1, y2, places=9)

    def test_stationary_collapses_to_zero(self):
        out = normalize_trajectory([(0.4, 0.4)] * 10)
        self.assertTrue(all(x == 0.0 and y == 0.0 for x, y in out))

    def test_empty(self):
        self.assertEqual(normalize_trajectory([]), [])


class TestResample(unittest.TestCase):

    def test_length(self):
        seq = [[float(i)] for i in range(10)]
        self.assertEqual(len(resample_sequence(seq, 24)), 24)
        self.assertEqual(len(resample_sequence(seq, 4)), 4)

    def test_endpoints_preserved(self):
        seq = [[0.0], [1.0], [2.0], [3.0]]
        out = resample_sequence(seq, 7)
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[-1][0], 3.0)

    def test_linear_values(self):
        seq = [[0.0], [10.0]]
        out = resample_sequence(seq, 5)
        expected = [0.0, 2.5, 5.0, 7.5, 10.0]
        for fr, e in zip(out, expected):
            self.assertAlmostEqual(fr[0], e)

    def test_single_frame_repeats(self):
        out = resample_sequence([[1.0, 2.0]], 5)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[3], [1.0, 2.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            resample_sequence([], 5)


# ---------------------------------------------------------------------------
# static classifier
# ---------------------------------------------------------------------------

class TestStaticClassifier(unittest.TestCase):

    def _trained(self):
        clf = StaticSignClassifier(k=3, unknown_threshold=1.8)
        for i in range(6):
            clf.add_sample(
                combine_hands(None, make_hand(curl=0.0, seed=i)), "OPEN")
            clf.add_sample(
                combine_hands(None, make_hand(curl=1.0, seed=100 + i)), "FIST")
            clf.add_sample(
                combine_hands(None, make_hand(spread=0.3, seed=200 + i)),
                "CLOSED_SPREAD")
        return clf

    def test_classifies_trained_poses(self):
        clf = self._trained()
        label, conf = clf.classify(
            combine_hands(None, make_hand(curl=0.0, seed=999)))
        self.assertEqual(label, "OPEN")
        # Note: synthetic hands are tiny (wrist->MCP = 0.05), so scale
        # normalization amplifies jitter ~20x vs real MediaPipe coords.
        # Confidence is therefore low here; label correctness is the test.
        self.assertGreater(conf, 0.2)
        label, conf = clf.classify(
            combine_hands(None, make_hand(curl=1.0, seed=998)))
        self.assertEqual(label, "FIST")

    def test_rotated_query_still_matches(self):
        clf = self._trained()
        h = transform_hand(make_hand(curl=1.0, seed=42), angle=0.9, scale=1.5)
        label, _ = clf.classify(combine_hands(None, h))
        self.assertEqual(label, "FIST")

    def test_empty_classifier_returns_none(self):
        clf = StaticSignClassifier()
        label, conf = clf.classify(combine_hands(None, make_hand()))
        self.assertIsNone(label)
        self.assertEqual(conf, 0.0)

    def test_hand_count_separation(self):
        clf = StaticSignClassifier()
        for i in range(4):
            clf.add_sample(
                combine_hands(make_hand(seed=i), make_hand(seed=50 + i)),
                "TWOHAND")
        # one-handed query must not match the two-handed class
        label, _ = clf.classify(combine_hands(None, make_hand(seed=7)))
        self.assertIsNone(label)

    def test_unknown_rejection(self):
        clf = StaticSignClassifier(unknown_threshold=0.05)  # very strict
        clf.add_sample(combine_hands(None, make_hand(curl=0.0)), "OPEN")
        clf.add_sample(combine_hands(None, make_hand(curl=0.0, seed=1)), "OPEN")
        clf.add_sample(combine_hands(None, make_hand(curl=0.0, seed=2)), "OPEN")
        label, conf = clf.classify(combine_hands(None, make_hand(curl=1.0)))
        self.assertIsNone(label)

    def test_remove_label(self):
        clf = self._trained()
        removed = clf.remove_label("FIST")
        self.assertEqual(removed, 6)
        self.assertNotIn("FIST", clf.label_counts())

    def test_bad_dimension_raises(self):
        clf = StaticSignClassifier()
        with self.assertRaises(ValueError):
            clf.add_sample([0.0] * 10, "X")

    def test_persistence_roundtrip(self):
        clf = self._trained()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "static.json")
            clf.save(path)
            clf2 = StaticSignClassifier.load(path)
        self.assertEqual(clf.label_counts(), clf2.label_counts())
        q = combine_hands(None, make_hand(curl=0.0, seed=77))
        self.assertEqual(clf.classify(q), clf2.classify(q))

    def test_load_rejects_bad_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                f.write('{"format": "something-else"}')
            with self.assertRaises(ValueError):
                StaticSignClassifier.load(path)


# ---------------------------------------------------------------------------
# dynamic classifier
# ---------------------------------------------------------------------------

def synth_recording(path_fn, n_frames=30, curl=0.0, seed=None):
    """Recording whose wrist follows path_fn(t), t in [0,1]."""
    rng = random.Random(seed) if seed is not None else None
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        x, y = path_fn(t)
        if rng:
            x += rng.uniform(-0.005, 0.005)
            y += rng.uniform(-0.005, 0.005)
        frames.append({"left": None,
                       "right": make_hand(curl=curl, seed=None),
                       "wrist": (x, y)})
    return frames


def circle_path(t):
    a = 2 * math.pi * t
    return 0.5 + 0.2 * math.cos(a), 0.5 + 0.2 * math.sin(a)


def swipe_path(t):
    return 0.2 + 0.6 * t, 0.5


def zigzag_path(t):
    return 0.2 + 0.6 * t, 0.5 + 0.15 * math.sin(6 * math.pi * t)


class TestDynamicClassifier(unittest.TestCase):

    def test_sequence_shape(self):
        seq = build_sequence_features(synth_recording(circle_path))
        self.assertEqual(len(seq), SEQ_LEN)
        self.assertEqual(len(seq[0]), FRAME_DIM)

    def test_dtw_zero_for_identical(self):
        seq = build_sequence_features(synth_recording(swipe_path))
        self.assertAlmostEqual(dtw_distance(seq, seq), 0.0, places=9)

    def test_dtw_speed_tolerance(self):
        fast = build_sequence_features(synth_recording(circle_path,
                                                       n_frames=15))
        slow = build_sequence_features(synth_recording(circle_path,
                                                       n_frames=60))
        diff = build_sequence_features(synth_recording(swipe_path,
                                                       n_frames=30))
        self.assertLess(dtw_distance(fast, slow),
                        dtw_distance(fast, diff))

    def test_classifies_paths(self):
        clf = DynamicSignClassifier(unknown_threshold=1.1)
        for i in range(5):
            clf.add_sample(build_sequence_features(
                synth_recording(circle_path, seed=i)), "CIRCLE")
            clf.add_sample(build_sequence_features(
                synth_recording(swipe_path, seed=100 + i)), "SWIPE")
            clf.add_sample(build_sequence_features(
                synth_recording(zigzag_path, seed=200 + i)), "ZIGZAG")

        for path, expect in ((circle_path, "CIRCLE"),
                             (swipe_path, "SWIPE"),
                             (zigzag_path, "ZIGZAG")):
            label, conf = clf.classify(build_sequence_features(
                synth_recording(path, seed=999)))
            self.assertEqual(label, expect)
            self.assertGreater(conf, 0.3)

    def test_pose_distinguishes_same_path(self):
        """Same trajectory, different handshape -> different signs."""
        clf = DynamicSignClassifier()
        for i in range(4):
            clf.add_sample(build_sequence_features(
                synth_recording(swipe_path, curl=0.0, seed=i)), "OPEN_SWIPE")
            clf.add_sample(build_sequence_features(
                synth_recording(swipe_path, curl=1.0, seed=50 + i)),
                "FIST_SWIPE")
        label, _ = clf.classify(build_sequence_features(
            synth_recording(swipe_path, curl=1.0, seed=888)))
        self.assertEqual(label, "FIST_SWIPE")

    def test_unknown_rejection(self):
        clf = DynamicSignClassifier(unknown_threshold=0.01)
        clf.add_sample(build_sequence_features(
            synth_recording(circle_path)), "CIRCLE")
        label, conf = clf.classify(build_sequence_features(
            synth_recording(zigzag_path)))
        self.assertIsNone(label)

    def test_empty_returns_none(self):
        clf = DynamicSignClassifier()
        label, conf = clf.classify(build_sequence_features(
            synth_recording(circle_path)))
        self.assertIsNone(label)

    def test_persistence_roundtrip(self):
        clf = DynamicSignClassifier()
        for i in range(3):
            clf.add_sample(build_sequence_features(
                synth_recording(circle_path, seed=i)), "CIRCLE")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dyn.json")
            clf.save(path)
            clf2 = DynamicSignClassifier.load(path)
        self.assertEqual(clf.label_counts(), clf2.label_counts())

    def test_short_recording_still_builds(self):
        seq = build_sequence_features(synth_recording(swipe_path, n_frames=8))
        self.assertEqual(len(seq), SEQ_LEN)

    def test_empty_recording_raises(self):
        with self.assertRaises(ValueError):
            build_sequence_features([])


# ---------------------------------------------------------------------------
# speller
# ---------------------------------------------------------------------------

class TestSpeller(unittest.TestCase):

    def _feed(self, sp, label, n, hand=True, motion=0.0):
        evs = []
        for _ in range(n):
            evs.extend(sp.update(label, hand, motion))
        return evs

    def test_letter_commits_after_hold(self):
        sp = Speller()
        evs = self._feed(sp, "A", 15)
        self.assertIn(("letter", "A"), evs)
        self.assertEqual(sp.current_word, ["A"])

    def test_flicker_does_not_commit(self):
        sp = Speller()
        evs = []
        for i in range(30):  # alternate every frame -- never stable
            evs.extend(sp.update("A" if i % 2 == 0 else "B", True))
        self.assertEqual(evs, [])

    def test_none_frames_block_commit(self):
        sp = Speller()
        evs = []
        for i in range(40):
            evs.extend(sp.update("A" if i % 3 else None, True))
        # front-runner run keeps resetting -> no commit
        self.assertEqual([e for e in evs if e[0] == "letter"], [])

    def test_no_immediate_repeat(self):
        sp = Speller()
        evs = self._feed(sp, "L", 60)  # long steady hold
        letters = [e for e in evs if e[0] == "letter"]
        self.assertEqual(letters, [("letter", "L")])  # exactly one

    def test_double_letter_after_gap(self):
        sp = Speller()
        self._feed(sp, "L", 20)
        self._feed(sp, None, sp.gap_frames + 1)   # re-form break
        evs = self._feed(sp, "L", 20)
        self.assertIn(("letter", "L"), evs)
        self.assertEqual(sp.current_word, ["L", "L"])

    def test_double_letter_after_motion_spike(self):
        sp = Speller()
        self._feed(sp, "L", 20)
        sp.update("L", True, motion=0.5)  # hand re-formed
        evs = self._feed(sp, "L", 20)
        self.assertEqual(sp.current_word, ["L", "L"])

    def test_word_commit_on_hand_absence(self):
        sp = Speller()
        self._feed(sp, "H", 20)
        self._feed(sp, None, sp.gap_frames + 1)
        self._feed(sp, "I", 20)
        evs = self._feed(sp, None, sp.word_gap_frames, hand=False)
        self.assertIn(("word", "HI"), evs)
        self.assertEqual(sp.words, ["HI"])
        self.assertEqual(sp.current_word, [])

    def test_text_rendering(self):
        sp = Speller()
        sp.words = ["HELLO"]
        sp.current_word = ["W", "O"]
        self.assertEqual(sp.text(), "HELLO WO")

    def test_backspace_and_clear(self):
        sp = Speller()
        sp.current_word = ["A", "B"]
        sp.backspace()
        self.assertEqual(sp.current_word, ["A"])
        sp.clear()
        self.assertEqual(sp.current_word, [])

    def test_fps_rescaling(self):
        sp = Speller()
        sp.report_fps(50.0)  # 2x base
        self.assertEqual(sp.hold_frames, 24)
        self.assertEqual(sp.word_gap_frames, 60)
        sp.report_fps(0)  # ignored
        self.assertEqual(sp.hold_frames, 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
