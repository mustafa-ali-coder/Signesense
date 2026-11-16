"""
speller.py -- turns a noisy stream of per-frame letter predictions into
committed letters and words.

Rules (all tunable):
  * A letter commits when it wins >= VOTE_NEED of the last VOTE_WINDOW frames
    AND has been the front-runner for >= hold_frames consecutive frames.
  * The same letter can't commit twice in a row until a "break" happens:
    either the prediction changes / drops to None for gap_frames, or the
    hand's motion energy spikes (the signer re-forms the letter). This is
    the standard double-letter problem in fingerspelling (e.g. "LL").
  * A word commits when no hand is seen for word_gap_frames.

Frame-rate note: defaults assume ~20-30 FPS. Feed report_fps() your real
FPS and it rescales.

Pure Python, no dependencies.
"""

from collections import deque


class Speller:

    def __init__(self,
                 vote_window=10,
                 vote_need=7,
                 hold_frames=12,
                 gap_frames=6,
                 word_gap_frames=30,
                 motion_break=0.35):
        self.vote_window = vote_window
        self.vote_need = vote_need
        self.hold_frames = hold_frames
        self.gap_frames = gap_frames
        self.word_gap_frames = word_gap_frames
        self.motion_break = motion_break

        self._votes = deque(maxlen=vote_window)
        self._front = None          # current front-runner label
        self._front_run = 0         # consecutive frames as front-runner
        self._last_committed = None
        self._break_since_commit = True
        self._none_run = 0
        self._no_hand_run = 0

        self.current_word = []
        self.words = []

    # -- tuning ---------------------------------------------------------------

    def report_fps(self, fps, base_fps=25.0):
        """Rescale frame-count thresholds to the measured FPS."""
        if fps <= 0:
            return
        r = fps / base_fps
        self.hold_frames = max(3, int(round(12 * r)))
        self.gap_frames = max(2, int(round(6 * r)))
        self.word_gap_frames = max(8, int(round(30 * r)))

    # -- main entry -----------------------------------------------------------

    def update(self, label, hand_present, motion=0.0):
        """Feed one frame.

        label: predicted letter (str) or None.
        hand_present: bool.
        motion: mean landmark displacement since last frame (normalized
                units); used only for double-letter breaks.

        Returns list of events, each ("letter", ch) or ("word", str).
        """
        events = []

        # --- word boundary on hand absence ---
        if not hand_present:
            self._no_hand_run += 1
            self._none_run += 1
            self._votes.append(None)
            self._front = None
            self._front_run = 0
            if self._none_run >= self.gap_frames:
                self._break_since_commit = True
            if (self._no_hand_run == self.word_gap_frames
                    and self.current_word):
                word = "".join(self.current_word)
                self.words.append(word)
                self.current_word = []
                events.append(("word", word))
            return events

        self._no_hand_run = 0

        # --- motion spike counts as a re-form break ---
        if motion >= self.motion_break:
            self._break_since_commit = True

        self._votes.append(label)

        if label is None:
            self._none_run += 1
            if self._none_run >= self.gap_frames:
                self._break_since_commit = True
            self._front = None
            self._front_run = 0
            return events
        self._none_run = 0

        # front-runner tracking
        if label == self._front:
            self._front_run += 1
        else:
            if self._front is not None:
                # prediction changed -> that's a break too
                self._break_since_commit = True
            self._front = label
            self._front_run = 1

        # commit check
        vote_count = sum(1 for v in self._votes if v == label)
        can_repeat = (label != self._last_committed
                      or self._break_since_commit)

        if (vote_count >= self.vote_need
                and self._front_run >= self.hold_frames
                and can_repeat):
            self.current_word.append(label)
            events.append(("letter", label))
            self._last_committed = label
            self._break_since_commit = False
            self._front_run = 0  # require a fresh hold for the next commit

        return events

    # -- utilities -------------------------------------------------------------

    def backspace(self):
        if self.current_word:
            self.current_word.pop()

    def clear(self):
        self.current_word = []

    def text(self):
        done = " ".join(self.words)
        cur = "".join(self.current_word)
        if done and cur:
            return done + " " + cur
        return done or cur
