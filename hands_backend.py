"""
hands_backend.py -- MediaPipe compatibility layer.

MediaPipe removed the legacy `mp.solutions` API in recent versions (which
are what installs on Python 3.13+). This shim presents ONE interface and
picks whichever backend exists:

    backend = HandsBackend()
    left, right, wrist = backend.process(bgr_frame)   # landmarks or None
    backend.draw(bgr_frame)                            # overlay skeleton
    print(backend.name)                                # for the HUD

Tasks-API path auto-downloads hand_landmarker.task (~8 MB, one time) next
to this file.
"""

import os
import time
import urllib.request

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")

# hand skeleton edges (same as mediapipe HAND_CONNECTIONS)
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class HandsBackend:

    def __init__(self, max_hands=2, min_confidence=0.6):
        import mediapipe as mp
        self._mp = mp
        self._last = ([], [])  # (landmark lists, handedness labels)

        if hasattr(mp, "solutions"):
            self.name = "legacy"
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence)
        else:
            self.name = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._vision = vision
            if not os.path.exists(MODEL_PATH):
                print("downloading hand_landmarker.task (one time, ~8 MB)…")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence)
            self._landmarker = vision.HandLandmarker.create_from_options(
                opts)
            self._t0 = time.time()

    def process(self, bgr_frame):
        """Returns (left, right, primary_wrist).
        left/right: 21 (x, y, z) tuples or None. primary_wrist: (x, y) of
        the right hand's wrist (or left if no right), or None."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        hands, labels = [], []

        if self.name == "legacy":
            res = self._hands.process(rgb)
            if res.multi_hand_landmarks:
                for lm_list, handed in zip(res.multi_hand_landmarks,
                                           res.multi_handedness):
                    hands.append([(p.x, p.y, p.z)
                                  for p in lm_list.landmark])
                    labels.append(handed.classification[0].label)
        else:
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - self._t0) * 1000)
            res = self._landmarker.detect_for_video(mp_img, ts_ms)
            for lm_list, handed in zip(res.hand_landmarks, res.handedness):
                hands.append([(p.x, p.y, p.z) for p in lm_list])
                labels.append(handed[0].category_name)

        self._last = (hands, labels)

        left = right = None
        for lms, label in zip(hands, labels):
            if label == "Left":
                left = lms
            else:
                right = lms
        primary = right if right is not None else left
        wrist = (primary[0][0], primary[0][1]) if primary else None
        return left, right, wrist

    def draw(self, bgr_frame):
        """Draw skeletons for the hands from the last process() call."""
        h, w = bgr_frame.shape[:2]
        for lms in self._last[0]:
            pts = [(int(x * w), int(y * h)) for x, y, _ in lms]
            for a, b in _CONNECTIONS:
                cv2.line(bgr_frame, pts[a], pts[b], (0, 200, 0), 2)
            for p in pts:
                cv2.circle(bgr_frame, p, 3, (0, 0, 255), -1)
