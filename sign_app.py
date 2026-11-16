"""
sign_app.py -- SignSense  |  Full GUI with CustomTkinter
"""

import os, sys, time, math, threading
import cv2, numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    CTK = True
except ImportError:
    CTK = False

import tkinter as tk
from tkinter import font as tkfont, messagebox, simpledialog

from hands_backend import HandsBackend
from sign_core import combine_hands
from static_sign import StaticSignClassifier
from dynamic_sign import DynamicSignClassifier, build_sequence_features
from speller import Speller
from record_signs import open_camera, STATIC_PATH, DYNAMIC_PATH
from speech import TTS, Autocorrect

C = dict(
    bg="#0d0d0d", panel="#141414", card="#1e1e1e", border="#2a2a2a",
    accent="#00bcd4", spell="#2196f3", sign_c="#f44336", green="#4caf50",
    yellow="#ffc107", red="#f44336", rec="#e53935", white="#ffffff",
    text="#e0e0e0", dim="#555555", sub="#888888",
)


def hex_to_bgr(h):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return b,g,r


def _Frame(parent, **kw):
    kw.setdefault("bg", C["panel"])
    return tk.Frame(parent, **kw)


def _Label(parent, **kw):
    kw.setdefault("bg", C["panel"])
    kw.setdefault("fg", C["text"])
    return tk.Label(parent, **kw)


def _Button(parent, text, cmd, color=None, fg=None, **kw):
    color = color or C["card"]; fg = fg or C["text"]
    f = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    b = tk.Label(parent, text=text, bg=color, fg=fg, font=f,
                 cursor="hand2", padx=10, pady=6, **kw)
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.configure(bg=_lighten(color)))
    b.bind("<Leave>",    lambda e: b.configure(bg=color))
    return b


def _lighten(hex_color, amount=20):
    h = hex_color.lstrip("#")
    try:
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"#{min(255,r+amount):02x}{min(255,g+amount):02x}{min(255,b+amount):02x}"
    except Exception:
        return hex_color


def _canvas_rect_alpha(frame, x, y, w, h, color_bgr, alpha=0.6):
    sub = frame[y:y+h, x:x+w]
    rect = np.full_like(sub, color_bgr, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)


class App:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SignSense  •  AI Sign Language Recognition")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1280x780")
        self.root.minsize(1024, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self._fonts()
        self._load_models()
        self._init_state()
        self._build_ui()
        self._bind_keys()
        self._camera_loop()
        self.root.mainloop()

    def _fonts(self):
        self.fnt = {
            "giant" : tkfont.Font(family="Segoe UI", size=64, weight="bold"),
            "big"   : tkfont.Font(family="Segoe UI", size=22, weight="bold"),
            "med"   : tkfont.Font(family="Segoe UI", size=14),
            "body"  : tkfont.Font(family="Consolas", size=16),
            "small" : tkfont.Font(family="Segoe UI", size=10),
            "tiny"  : tkfont.Font(family="Segoe UI", size=8),
            "btn"   : tkfont.Font(family="Segoe UI", size=10, weight="bold"),
            "badge" : tkfont.Font(family="Segoe UI", size=11, weight="bold"),
        }

    def _load_models(self):
        self.static_clf  = (StaticSignClassifier.load(STATIC_PATH)
                            if os.path.exists(STATIC_PATH) else None)
        self.dynamic_clf = (DynamicSignClassifier.load(DYNAMIC_PATH)
                            if os.path.exists(DYNAMIC_PATH) else None)
        if not self.static_clf and not self.dynamic_clf:
            messagebox.showerror("SignSense",
                "No datasets found.\nRun: python run.py record static")
            sys.exit(1)
        self.tts = TTS()
        self.ac  = Autocorrect()
        self.backend = HandsBackend()
        self.cap = open_camera()

    def _init_state(self):
        self.mode         = "SPELL" if self.static_clf else "SIGN"
        self.speller      = Speller()
        self.recording    = False
        self.rec_frames   = []
        self.last_sign    = ""
        self.last_conf    = 0.0
        self.pending_seq  = None   # unknown sequence waiting for label
        self.prev_primary = None
        self.cur_label    = None
        self.cur_conf     = 0.0
        self.hand_present = False
        self.fps          = 0.0
        self.fps_t0       = time.time()
        self.fps_n        = 0
        self.fps_done     = False
        self.frame_count  = 0
        self.running      = True
        self.fullscreen   = False
        self.tts_on       = True
        self.ac_on        = True
        self.word_log               = []
        self._photo                 = None
        self._flash                 = 0.0
        self._autosave_static_pending = False
        self._autosave_counter      = 0
        self._AUTOSAVE_EVERY        = 20

    # ── UI BUILD ──────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_titlebar()
        main = _Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=(0,10))
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=2, minsize=320)
        main.rowconfigure(0, weight=1)
        self._build_camera(main)
        self._build_side(main)
        self._build_textbar()

    def _build_titlebar(self):
        bar = _Frame(self.root, bg=C["panel"], height=52)
        bar.pack(fill="x", padx=10, pady=(10,4))
        bar.pack_propagate(False)

        tk.Label(bar, text="◈  SignSense", bg=C["panel"], fg=C["accent"],
                 font=self.fnt["big"]).pack(side="left", padx=14)

        tabs = _Frame(bar, bg=C["panel"])
        tabs.pack(side="left", padx=20)
        self._spell_tab = self._make_tab(tabs, "SPELL", "SPELL")
        self._sign_tab  = self._make_tab(tabs, "SIGN",  "SIGN")
        self._refresh_tabs()

        right = _Frame(bar, bg=C["panel"])
        right.pack(side="right", padx=10)

        self._fs_btn = _Button(right, "⛶  Fullscreen",
                               self._toggle_fullscreen,
                               color=C["card"], fg=C["sub"])
        self._fs_btn.pack(side="right", padx=4)

        self._tts_btn = _Button(right,
                                "🔊 Speech ON" if self.tts_on else "🔇 Speech OFF",
                                self._toggle_tts,
                                color=C["green"] if self.tts_on else C["card"],
                                fg=C["white"])
        self._tts_btn.pack(side="right", padx=4)

        self._ac_btn = _Button(right,
                               "✓ Autocorrect" if self.ac_on else "✗ Autocorrect",
                               self._toggle_ac,
                               color=C["spell"] if self.ac_on else C["card"],
                               fg=C["white"])
        self._ac_btn.pack(side="right", padx=4)

        info = _Frame(right, bg=C["panel"])
        info.pack(side="right", padx=12)
        self._fps_lbl = _Label(info, text="-- FPS", fg=C["dim"],
                               font=self.fnt["small"])
        self._fps_lbl.pack(anchor="e")
        _Label(info, text=self.backend.name.upper(), fg=C["dim"],
               font=self.fnt["tiny"]).pack(anchor="e")

        self._dot_canvas = tk.Canvas(right, width=16, height=16,
                                     bg=C["panel"], highlightthickness=0)
        self._dot_canvas.pack(side="right")
        self._dot = self._dot_canvas.create_oval(3,3,13,13,
                                                  fill=C["dim"], outline="")

    def _make_tab(self, parent, mode, text):
        lbl = tk.Label(parent, text=f"  {text}  ", font=self.fnt["badge"],
                       cursor="hand2", padx=6, pady=4)
        lbl.bind("<Button-1>", lambda e, m=mode: self._set_mode(m))
        lbl.pack(side="left", padx=2)
        return lbl

    def _refresh_tabs(self):
        if self.mode == "SPELL":
            self._spell_tab.configure(bg=C["spell"], fg=C["white"])
            self._sign_tab.configure( bg=C["card"],  fg=C["dim"])
        else:
            self._spell_tab.configure(bg=C["card"],  fg=C["dim"])
            self._sign_tab.configure( bg=C["sign_c"],fg=C["white"])

    def _build_camera(self, parent):
        wrap = _Frame(parent, bg=C["border"], bd=1, relief="solid")
        wrap.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self._cam_canvas = tk.Canvas(wrap, bg=C["bg"], highlightthickness=0)
        self._cam_canvas.pack(fill="both", expand=True)

    def _build_side(self, parent):
        side = _Frame(parent, bg=C["panel"])
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        self._build_letter_card(side)
        self._build_sign_card(side)
        self._build_vocab_card(side)
        self._build_log_card(side)

    def _card(self, parent, title, row):
        outer = _Frame(parent, bg=C["border"], bd=1, relief="solid")
        outer.grid(row=row, column=0, sticky="ew", padx=6, pady=(0,6))
        _Label(outer, text=title, fg=C["dim"],
               font=self.fnt["tiny"], bg=C["card"]).pack(
                   fill="x", padx=8, pady=(4,0), anchor="w")
        body = _Frame(outer, bg=C["card"])
        body.pack(fill="x", padx=4, pady=(0,4))
        return body

    def _build_letter_card(self, parent):
        body = self._card(parent, "DETECTED LETTER", 0)
        row = _Frame(body, bg=C["card"])
        row.pack(fill="x", padx=8, pady=8)
        self._letter_lbl = tk.Label(row, text="-", bg=C["bg"],
                                    fg=C["dim"], font=self.fnt["giant"],
                                    width=2, relief="flat")
        self._letter_lbl.pack(side="left", padx=(0,12))
        right = _Frame(row, bg=C["card"])
        right.pack(side="left", fill="both", expand=True)
        _Label(right, text="CONFIDENCE", fg=C["dim"], bg=C["card"],
               font=self.fnt["tiny"]).pack(anchor="w")
        self._conf_canvas = tk.Canvas(right, height=16, bg=C["bg"],
                                      highlightthickness=0)
        self._conf_canvas.pack(fill="x", pady=(2,2))
        self._conf_bar = self._conf_canvas.create_rectangle(
            0, 0, 0, 16, fill=C["green"])
        self._conf_pct = _Label(right, text="0%", fg=C["dim"],
                                bg=C["card"], font=self.fnt["small"])
        self._conf_pct.pack(anchor="w")

    def _build_sign_card(self, parent):
        body = self._card(parent, "SIGN MODE", 1)

        self._rec_btn = _Button(body, "  ◉  RECORD  (R)  ",
                                self._toggle_record,
                                color=C["card"], fg=C["white"])
        self._rec_btn.pack(fill="x", padx=8, pady=(8,4))

        self._rec_status = _Label(body, text="Press R to record a sign",
                                  fg=C["dim"], bg=C["card"],
                                  font=self.fnt["small"])
        self._rec_status.pack(anchor="w", padx=8)

        self._sign_result = _Label(body, text="", fg=C["accent"],
                                   bg=C["card"], font=self.fnt["med"])
        self._sign_result.pack(anchor="w", padx=8, pady=(4,0))

        # ── unknown recording actions ──────────────────────────────────
        self._unknown_frame = _Frame(body, bg=C["card"])
        self._unknown_frame.pack(fill="x", padx=8, pady=(4,8))

        self._save_unknown_btn = _Button(
            self._unknown_frame,
            "  💾  Label & Save  ",
            self._save_unknown,
            color=C["yellow"], fg="#000000")
        self._save_unknown_btn.pack(side="left", padx=(0,4))

        self._discard_btn = _Button(
            self._unknown_frame,
            "  ✕  Discard  ",
            self._discard_unknown,
            color=C["card"], fg=C["sub"])
        self._discard_btn.pack(side="left")

        # hidden until needed
        self._unknown_frame.pack_forget()

    def _build_vocab_card(self, parent):
        body = self._card(parent, "VOCABULARY", 2)
        parts = []
        if self.static_clf:
            L = sorted(self.static_clf.label_counts())
            parts.append(f"Letters ({len(L)}):  " + " ".join(L))
        if self.dynamic_clf:
            W = sorted(self.dynamic_clf.label_counts())
            parts.append(f"Signs ({len(W)}):  " + ", ".join(W))
        self._vocab_lbl = _Label(body,
               text="\n".join(parts) or "—",
               fg=C["sub"], bg=C["card"],
               font=self.fnt["tiny"], justify="left",
               wraplength=290)
        self._vocab_lbl.pack(anchor="w", padx=8, pady=6)

    def _refresh_vocab(self):
        parts = []
        if self.static_clf:
            L = sorted(self.static_clf.label_counts())
            parts.append(f"Letters ({len(L)}):  " + " ".join(L))
        if self.dynamic_clf:
            W = sorted(self.dynamic_clf.label_counts())
            parts.append(f"Signs ({len(W)}):  " + ", ".join(W))
        self._vocab_lbl.configure(text="\n".join(parts) or "—")

    def _build_log_card(self, parent):
        outer = _Frame(parent, bg=C["border"], bd=1, relief="solid")
        outer.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0,6))
        parent.rowconfigure(3, weight=1)
        _Label(outer, text="WORD LOG", fg=C["dim"],
               font=self.fnt["tiny"], bg=C["card"]).pack(
                   fill="x", padx=8, pady=(4,0), anchor="w")
        self._log = tk.Text(outer, bg=C["card"], fg=C["text"],
                            font=self.fnt["small"], wrap="word",
                            state="disabled", relief="flat",
                            highlightthickness=0, padx=8, pady=4)
        self._log.pack(fill="both", expand=True, padx=4, pady=(0,4))

    def _build_textbar(self):
        bar = _Frame(self.root, bg=C["card"], height=56)
        bar.pack(fill="x", padx=10, pady=(0,10))
        bar.pack_propagate(False)
        _Label(bar, text="TEXT", fg=C["dim"], bg=C["card"],
               font=self.fnt["tiny"]).pack(side="left", padx=(12,6),
                                            anchor="center")
        self._text_lbl = tk.Label(bar, text="▌", bg=C["bg"], fg=C["white"],
                                  font=self.fnt["body"], anchor="w",
                                  padx=12, relief="flat")
        self._text_lbl.pack(side="left", fill="both", expand=True,
                             padx=(0,8), pady=8)
        btns = _Frame(bar, bg=C["card"])
        btns.pack(side="right", padx=8)
        _Button(btns, "▶  SPEAK",    self._speak_current,
                color=C["panel"], fg=C["accent"]).pack(side="right", padx=2)
        _Button(btns, "⌫  DEL",     self._backspace,
                color=C["card"], fg=C["sub"]).pack(side="right", padx=2)
        _Button(btns, "✕  CLEAR",   self._clear,
                color=C["card"], fg=C["sub"]).pack(side="right", padx=2)

    def _bind_keys(self):
        r = self.root
        r.bind("<Tab>",       lambda e: self._toggle_mode())
        r.bind("<BackSpace>", lambda e: self._backspace())
        r.bind("<Key-c>",     lambda e: self._clear())
        r.bind("<Key-C>",     lambda e: self._clear())
        r.bind("<Key-q>",     lambda e: self._quit())
        r.bind("<Key-Q>",     lambda e: self._quit())
        r.bind("<Key-r>",     lambda e: self._toggle_record())
        r.bind("<Key-R>",     lambda e: self._toggle_record())
        r.bind("<F11>",       lambda e: self._toggle_fullscreen())
        r.bind("<Escape>",    lambda e: self._exit_fullscreen())

    # ── mode ──────────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        if mode == "SPELL" and not self.static_clf:  return
        if mode == "SIGN"  and not self.dynamic_clf: return
        self.mode = mode
        self.recording = False
        self.rec_frames = []
        self._hide_unknown_actions()
        self._refresh_tabs()

    def _toggle_mode(self):
        self._set_mode("SIGN" if self.mode == "SPELL" else "SPELL")

    # ── recording ─────────────────────────────────────────────────────────

    def _toggle_record(self):
        if self.mode != "SIGN" or not self.dynamic_clf:
            return
        # if unknown actions are showing, ignore R until resolved
        if self.pending_seq is not None:
            return
        if self.recording:
            self.recording = False
            self._rec_btn.configure(text="  ◉  RECORD  (R)  ",
                                    bg=C["card"])
            if len(self.rec_frames) >= 8:
                seq = build_sequence_features(self.rec_frames)
                w, c = self.dynamic_clf.classify(seq)
                if w:
                    # ── known sign -- auto-save for training ────────────
                    self.last_sign = w
                    self.last_conf = c
                    self._sign_result.configure(
                        text=f"{w.upper()}   ({c:.0%})",
                        fg=C["green"] if c > 0.55 else C["yellow"])
                    self._log_word(f"[sign] {w}")
                    if self.tts_on:
                        self.tts.speak(w)
                    self.dynamic_clf.add_sample(seq, w)
                    self.dynamic_clf.save(DYNAMIC_PATH)
                    self._hide_unknown_actions()
                else:
                    # ── unknown sign → offer to label & save ────────────
                    self.pending_seq = seq
                    self._sign_result.configure(
                        text="Unknown — label it to train?",
                        fg=C["yellow"])
                    self._show_unknown_actions()
                self._rec_status.configure(
                    text="Press R to record again",
                    fg=C["dim"])
            else:
                self._rec_status.configure(
                    text="Too short — hold longer", fg=C["red"])
                self.root.after(2000, lambda: self._rec_status.configure(
                    text="Press R to record a sign", fg=C["dim"]))
                self._hide_unknown_actions()
            self.rec_frames = []
        else:
            self.recording = True
            self.rec_frames = []
            self.pending_seq = None
            self._hide_unknown_actions()
            self._rec_btn.configure(text="  ■  STOP  (R)  ", bg=C["rec"])
            self._rec_status.configure(text="Recording...", fg=C["rec"])
            self._sign_result.configure(text="")

    def _show_unknown_actions(self):
        self._unknown_frame.pack(fill="x", padx=8, pady=(4,8))

    def _hide_unknown_actions(self):
        self._unknown_frame.pack_forget()
        self.pending_seq = None

    def _save_unknown(self):
        """Label the pending unknown sequence and add it to the dataset."""
        if self.pending_seq is None:
            return
        label = simpledialog.askstring(
            "Label Recording",
            "Enter a label for this sign\n(e.g. 'hello', 'J', 'stop'):",
            parent=self.root)
        if label and label.strip():
            label = label.strip()
            self.dynamic_clf.add_sample(self.pending_seq, label)
            self.dynamic_clf.save(DYNAMIC_PATH)
            self._sign_result.configure(
                text=f"Saved as  '{label}'  ✓",
                fg=C["green"])
            self._log_word(f"[trained] {label}")
            self._refresh_vocab()
            self._hide_unknown_actions()
        else:
            # cancelled -- keep showing the buttons
            pass

    def _discard_unknown(self):
        """Throw away the pending unknown sequence."""
        self._sign_result.configure(text="Discarded", fg=C["dim"])
        self._hide_unknown_actions()
        self.root.after(1500, lambda: self._sign_result.configure(text=""))

    # ── speller / text ────────────────────────────────────────────────────

    def _backspace(self):
        self.speller.backspace()

    def _clear(self):
        self.speller.clear()
        self.word_log.clear()
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _speak_current(self):
        text = self.speller.text().strip()
        if text and self.tts.available:
            self.tts.speak_force(text)

    def _log_word(self, word):
        self.word_log.append(word)
        self._log.configure(state="normal")
        self._log.insert("end", word + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    # ── settings ──────────────────────────────────────────────────────────

    def _toggle_tts(self):
        self.tts_on = not self.tts_on
        if self.tts_on:
            self._tts_btn.configure(text="🔊 Speech ON",
                                    bg=C["green"], fg=C["white"])
        else:
            self._tts_btn.configure(text="🔇 Speech OFF",
                                    bg=C["card"], fg=C["sub"])

    def _toggle_ac(self):
        self.ac_on = not self.ac_on
        if self.ac_on:
            self._ac_btn.configure(text="✓ Autocorrect",
                                   bg=C["spell"], fg=C["white"])
        else:
            self._ac_btn.configure(text="✗ Autocorrect",
                                   bg=C["card"], fg=C["sub"])

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        lbl = "✕  Exit Fullscreen" if self.fullscreen else "⛶  Fullscreen"
        self._fs_btn.configure(text=lbl)

    def _exit_fullscreen(self):
        if self.fullscreen:
            self._toggle_fullscreen()

    # ── camera loop ───────────────────────────────────────────────────────

    def _camera_loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._camera_loop)
            return

        frame = cv2.flip(frame, 1)
        left, right, wrist = self.backend.process(frame)
        primary = right if right is not None else left
        self.backend.draw(frame)
        self.hand_present = primary is not None

        self.fps_n += 1
        self.frame_count += 1
        dt = time.time() - self.fps_t0
        if dt >= 1.0:
            self.fps = self.fps_n / dt
            self.fps_t0, self.fps_n = time.time(), 0
            if not self.fps_done and self.fps > 5:
                self.speller.report_fps(self.fps)
                self.fps_done = True

        self.cur_label, self.cur_conf = None, 0.0

        if self.mode == "SPELL" and self.static_clf:
            if self.hand_present:
                feats = combine_hands(left, right)
                self.cur_label, self.cur_conf = \
                    self.static_clf.classify(feats)
            motion = self._motion(self.prev_primary, primary)
            for ev, val in self.speller.update(
                    self.cur_label, self.hand_present, motion):
                if ev == "word":
                    word = val
                    if self.ac_on and self.ac.available:
                        corrected = self.ac.correct(word)
                        if corrected.lower() != word.lower():
                            word = corrected
                    self._log_word(word)
                    if self.tts_on and self.tts.available:
                        self.tts.speak(word)
                elif ev == "letter" and self.hand_present and self.static_clf:
                    # auto-save: committed letters reinforce the model
                    if left is not None or right is not None:
                        try:
                            feats = combine_hands(left, right)
                            self.static_clf.add_sample(feats, val)
                            self._autosave_static_pending = True
                        except Exception:
                            pass

        elif self.mode == "SIGN" and self.dynamic_clf:
            if self.recording and wrist:
                self.rec_frames.append(
                    {"left": left, "right": right, "wrist": wrist})
                self._rec_status.configure(
                    text=f"Recording...  {len(self.rec_frames)} frames")

        self.prev_primary = primary

        # flush static autosave periodically
        if self._autosave_static_pending:
            self._autosave_counter += 1
            if self._autosave_counter >= self._AUTOSAVE_EVERY:
                try:
                    self.static_clf.save(STATIC_PATH)
                except Exception:
                    pass
                self._autosave_counter = 0
                self._autosave_static_pending = False

        self._draw_overlays(frame)

        cw = self._cam_canvas.winfo_width()
        ch = self._cam_canvas.winfo_height()
        if cw > 10 and ch > 10:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((cw, ch), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self._cam_canvas.create_image(0, 0, anchor="nw",
                                          image=self._photo)

        self._update_widgets()
        self.root.after(1, self._camera_loop)

    def _draw_overlays(self, frame):
        h, w = frame.shape[:2]

        if self.recording:
            pulse = int(180 + 75 * math.sin(self.frame_count * 0.18))
            cv2.rectangle(frame, (0,0), (w-1,h-1),
                          (30, 30, pulse), 5)

        mode_color = hex_to_bgr(C["spell"] if self.mode=="SPELL"
                                 else C["sign_c"])
        cv2.rectangle(frame, (12,12), (110,40), mode_color, -1)
        cv2.rectangle(frame, (12,12), (110,40), (255,255,255), 1)
        cv2.putText(frame, self.mode, (22,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        fps_txt = f"{self.fps:.0f} FPS"
        tw = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(frame, fps_txt, (w-tw-10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    hex_to_bgr(C["dim"]), 1)

        if self.mode == "SPELL" and self.cur_label:
            c = self.cur_conf
            col = (hex_to_bgr(C["green"])  if c >= 0.70 else
                   hex_to_bgr(C["yellow"]) if c >= 0.40 else
                   hex_to_bgr(C["red"]))
            _canvas_rect_alpha(frame, 10, h-80, 80, 70,
                               hex_to_bgr(C["bg"]), 0.7)
            cv2.putText(frame, self.cur_label, (18, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, 4)
            bar_x, bar_y, bar_w = 95, h-60, 120
            cv2.rectangle(frame, (bar_x,bar_y),
                          (bar_x+bar_w, bar_y+10),
                          hex_to_bgr(C["card"]), -1)
            cv2.rectangle(frame, (bar_x,bar_y),
                          (bar_x+int(bar_w*c), bar_y+10), col, -1)
            cv2.putText(frame, f"{c:.0%}", (bar_x, bar_y+26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        hex_to_bgr(C["sub"]), 1)

        if self._flash > 0:
            overlay = np.full_like(frame, 255, dtype=np.uint8)
            cv2.addWeighted(overlay, self._flash*0.25,
                            frame, 1-self._flash*0.25, 0, frame)
            self._flash = max(0.0, self._flash - 0.12)

    def _update_widgets(self):
        self._fps_lbl.configure(text=f"{self.fps:.0f} FPS")
        self._dot_canvas.itemconfigure(
            self._dot,
            fill=C["green"] if self.hand_present else C["dim"])

        if self.cur_label:
            c = self.cur_conf
            col = (C["green"] if c >= 0.70 else
                   C["yellow"] if c >= 0.40 else C["red"])
            self._letter_lbl.configure(text=self.cur_label,
                                       fg=col, bg=C["bg"])
            bw = self._conf_canvas.winfo_width()
            self._conf_canvas.coords(self._conf_bar, 0, 0, int(bw*c), 16)
            self._conf_canvas.itemconfigure(self._conf_bar, fill=col)
            self._conf_pct.configure(text=f"{c:.0%}", fg=col)
        else:
            self._letter_lbl.configure(text="-", fg=C["dim"])
            self._conf_canvas.coords(self._conf_bar, 0, 0, 0, 16)
            self._conf_pct.configure(text="0%", fg=C["dim"])

        text = self.speller.text()
        disp = text + "▌" if text else "▌"
        if len(disp) > 55:
            disp = "…" + disp[-52:]
        self._text_lbl.configure(text=disp)

    @staticmethod
    def _motion(prev, cur):
        if prev is None or cur is None:
            return 0.0
        s = sum(abs(a[0]-b[0]) + abs(a[1]-b[1])
                for a, b in zip(prev, cur))
        return s / len(cur)

    def _quit(self):
        self.running = False
        try:
            if self._autosave_static_pending and self.static_clf:
                self.static_clf.save(STATIC_PATH)
        except Exception:
            pass
        try:
            self.tts.shutdown()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    App()


if __name__ == "__main__":
    main()
