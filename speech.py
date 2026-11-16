"""
speech.py -- Text-to-speech and autocorrect for SignSense.

TTS: Uses Windows SAPI via a persistent PowerShell process with a queue.
     One background thread feeds words one at a time -- reliable, sequential,
     speaks every word without skipping or getting stuck.

Autocorrect: edit-distance correction against a bundled word list.
"""

import os
import json
import threading
import queue
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST_PATH = os.path.join(HERE, "wordlist.json")


# ── TTS ──────────────────────────────────────────────────────────────────────

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
while ($true) {
    $line = [Console]::ReadLine()
    if ($line -eq $null) { break }
    $line = $line.Trim()
    if ($line -eq 'EXIT') { break }
    if ($line.Length -gt 0) {
        $synth.Speak($line)
    }
}
$synth.Dispose()
"""


class TTS:
    """
    Non-blocking TTS using a persistent PowerShell SAPI process.
    Words are queued and spoken sequentially -- every word gets spoken.
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._queue: queue.Queue = queue.Queue()
        self._proc = None
        self._thread = None
        if sys.platform == "win32":
            self._start()

    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NonInteractive", "-Command", _PS_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
            self._enabled = True
            self._thread = threading.Thread(
                target=self._worker, daemon=True)
            self._thread.start()
        except Exception:
            self._enabled = False

    def _worker(self) -> None:
        """Background thread: dequeue words and send to PowerShell."""
        while True:
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                # check if process is still alive
                if self._proc and self._proc.poll() is not None:
                    break
                continue
            if text is None:   # shutdown signal
                break
            try:
                if self._proc and self._proc.stdin:
                    self._proc.stdin.write(text + "\n")
                    self._proc.stdin.flush()
            except Exception:
                break

    @property
    def available(self) -> bool:
        return self._enabled

    def speak(self, text: str) -> None:
        """Queue text for speaking. Returns immediately."""
        if not self._enabled or not text.strip():
            return
        safe = text.replace("'", "").replace('"', "").replace("`", "").strip()
        if safe:
            self._queue.put(safe)

    def speak_force(self, text: str) -> None:
        """Clear queue then speak -- use for on-demand replay."""
        if not self._enabled or not text.strip():
            return
        # drain existing queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.speak(text)

    def set_rate(self, wpm: int) -> None:
        # send rate change command (PowerShell script doesn't support live
        # rate changes in this simplified version -- restart would be needed)
        pass

    def set_volume(self, vol: float) -> None:
        pass

    def shutdown(self) -> None:
        if self._proc:
            try:
                self._queue.put(None)
                if self._proc.stdin:
                    self._proc.stdin.write("EXIT\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


# ── AUTOCORRECT ───────────────────────────────────────────────────────────────

def _edit1(word: str) -> set:
    letters = "abcdefghijklmnopqrstuvwxyz"
    w = word.lower()
    n = len(w)
    deletes    = {w[:i] + w[i+1:]            for i in range(n)}
    transposes = {w[:i]+w[i+1]+w[i]+w[i+2:] for i in range(n - 1)}
    replaces   = {w[:i] + c + w[i+1:]       for i in range(n) for c in letters}
    inserts    = {w[:i] + c + w[i:]         for i in range(n + 1) for c in letters}
    return deletes | transposes | replaces | inserts


PRIORITY: set = {
    "hello", "water", "drink", "eat", "please", "sorry", "friend", "good",
    "yes", "no", "help", "name", "love", "school", "book", "computer",
    "family", "learn", "thank", "you", "bad", "fine", "okay", "want",
    "need", "know", "like", "have", "come", "go", "home", "work", "time",
    "day", "food", "hand", "sign", "language", "pakistan", "english",
    "urdu", "people", "see", "hear", "speak", "understand", "again", "stop",
}


class Autocorrect:
    """Correct misspelled words using edit-distance against a word list."""

    def __init__(self) -> None:
        self._words: set = set()
        self._loaded: bool = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(WORDLIST_PATH):
            with open(WORDLIST_PATH, encoding="utf-8") as f:
                self._words = set(json.load(f))
            self._loaded = True

    @property
    def available(self) -> bool:
        return self._loaded

    @staticmethod
    def _recase(corrected: str, original: str) -> str:
        if original and original[0].isupper():
            return corrected.capitalize()
        return corrected

    def correct(self, word: str) -> str:
        if not self._loaded:
            return word
        w = word.strip().lower()
        if len(w) < 3 or w in self._words:
            return word
        e1 = {c for c in _edit1(w) if c in self._words}
        pri = e1 & PRIORITY
        if len(pri) == 1:
            return self._recase(next(iter(pri)), word)
        same = {c for c in e1 if len(c) == len(w)}
        if len(same) == 1:
            return self._recase(next(iter(same)), word)
        e2_pri: set = set()
        for mid in e1:
            for c in _edit1(mid):
                if c in self._words and c in PRIORITY:
                    e2_pri.add(c)
        e2_pri.discard(w)
        if len(e2_pri) == 1:
            return self._recase(next(iter(e2_pri)), word)
        return word

    def is_word(self, word: str) -> bool:
        return word.lower() in self._words

    def suggest(self, word: str, n: int = 3) -> list:
        if not self._loaded:
            return []
        w = word.strip().lower()
        cands = [c for c in _edit1(w) if c in self._words]
        cands.sort(key=lambda c: abs(len(c) - len(w)))
        return cands[:n]
