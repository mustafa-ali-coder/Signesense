"""
record_signs.py -- SignSense dataset recorder with full GUI.
"""

import os, sys, time, math
import tkinter as tk
from tkinter import font as tkfont, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np

from hands_backend import HandsBackend
from sign_core import combine_hands
from static_sign import StaticSignClassifier
from dynamic_sign import DynamicSignClassifier, build_sequence_features

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_PATH  = os.path.join(HERE, "static_signs.json")
DYNAMIC_PATH = os.path.join(HERE, "dynamic_signs.json")

C = dict(
    bg=    "#0d0d0d", panel="#141414", card="#1e1e1e",
    border="#2a2a2a", accent="#00bcd4", green="#4caf50",
    yellow="#ffc107", red="#f44336",   white="#ffffff",
    text=  "#e0e0e0", dim="#555555",   sub="#888888",
    rec=   "#e53935",
)

def open_camera():
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("no webcam found")


def _btn(parent, text, cmd, bg=None, fg=None):
    bg = bg or C["card"]; fg = fg or C["text"]
    f = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    b = tk.Label(parent, text=text, bg=bg, fg=fg, font=f,
                 cursor="hand2", padx=10, pady=6)
    b.bind("<Button-1>", lambda e: cmd())
    return b


class RecorderApp:
    def __init__(self, mode="static"):
        self.mode = mode
        self.root = tk.Tk()
        self.root.title(f"SignSense Recorder — {mode.upper()}")
        self.root.configure(bg=C["bg"])
        self.root.geometry("980x660")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.fnt = {
            "big":   tkfont.Font(family="Segoe UI", size=18, weight="bold"),
            "giant": tkfont.Font(family="Segoe UI", size=48, weight="bold"),
            "med":   tkfont.Font(family="Segoe UI", size=12),
            "small": tkfont.Font(family="Segoe UI", size=9),
            "btn":   tkfont.Font(family="Segoe UI", size=10, weight="bold"),
        }

        self.clf = (StaticSignClassifier.load(STATIC_PATH)
                    if mode=="static" and os.path.exists(STATIC_PATH)
                    else StaticSignClassifier()) if mode=="static" else \
                   (DynamicSignClassifier.load(DYNAMIC_PATH)
                    if os.path.exists(DYNAMIC_PATH)
                    else DynamicSignClassifier())

        self.backend  = HandsBackend()
        self.cap      = open_camera()
        self.label    = None
        self.recording= False
        self.rec_frames=[]
        self.running  = True
        self.frame_count=0
        self.flash    = 0.0
        self._last_left=self._last_right=None
        self.TARGET = 12 if mode=="static" else 6

        self._build()
        self._ask_label()
        self._loop()
        self.root.mainloop()

    def _build(self):
        # title bar
        top = tk.Frame(self.root, bg=C["panel"], height=48)
        top.pack(fill="x", padx=8, pady=(8,4))
        top.pack_propagate(False)
        mode_col = C["accent"]
        tk.Label(top, text=f"◈  SignSense  |  {self.mode.upper()} RECORDER",
                 bg=C["panel"], fg=mode_col,
                 font=self.fnt["big"]).pack(side="left", padx=14)

        # main
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=8, pady=(0,8))
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1, minsize=280)
        main.rowconfigure(0, weight=1)

        # camera
        cam = tk.Frame(main, bg=C["border"], bd=1, relief="solid")
        cam.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self.canvas = tk.Canvas(cam, bg=C["bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # side
        side = tk.Frame(main, bg=C["panel"])
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        self._build_side(side)

        # bindings
        self.root.bind("<space>",  lambda e: self._capture())
        self.root.bind("<Key-n>",  lambda e: self._ask_label())
        self.root.bind("<Key-N>",  lambda e: self._ask_label())
        self.root.bind("<Key-q>",  lambda e: self._save_quit())
        self.root.bind("<Key-Q>",  lambda e: self._save_quit())
        self.root.bind("<Key-r>",  lambda e: self._toggle_rec())
        self.root.bind("<Key-R>",  lambda e: self._toggle_rec())

    def _build_side(self, side):
        def card(title, row):
            outer = tk.Frame(side, bg=C["border"], bd=1, relief="solid")
            outer.grid(row=row, column=0, sticky="ew", padx=6, pady=(0,6))
            tk.Label(outer, text=title, bg=C["card"], fg=C["dim"],
                     font=tkfont.Font(family="Segoe UI",size=8)).pack(
                         fill="x", padx=6, pady=(4,0), anchor="w")
            body = tk.Frame(outer, bg=C["card"])
            body.pack(fill="x", padx=4, pady=(0,4))
            return body

        # label card
        lc = card("CURRENT LABEL", 0)
        self.label_lbl = tk.Label(lc, text="-", bg=C["bg"], fg=C["accent"],
                                  font=self.fnt["giant"], width=3, height=1)
        self.label_lbl.pack(padx=8, pady=8)

        # progress card
        pc = card("PROGRESS", 1)
        self.count_lbl = tk.Label(pc, text="0 / " + str(self.TARGET),
                                  bg=C["card"], fg=C["green"],
                                  font=self.fnt["med"])
        self.count_lbl.pack(anchor="w", padx=8, pady=(6,2))
        self.prog_canvas = tk.Canvas(pc, height=14, bg=C["bg"],
                                     highlightthickness=0)
        self.prog_canvas.pack(fill="x", padx=8, pady=(0,6))
        self.prog_bar = self.prog_canvas.create_rectangle(
            0, 0, 0, 14, fill=C["green"])

        # action card
        ac = card("ACTIONS", 2)
        if self.mode == "static":
            _btn(ac, "  CAPTURE  (SPACE)  ",
                 self._capture, bg=C["green"], fg=C["white"]).pack(
                     fill="x", padx=8, pady=(8,4))
        else:
            self.rec_btn = _btn(ac, "  ◉  RECORD  (R)  ",
                                self._toggle_rec, bg=C["card"])
            self.rec_btn.pack(fill="x", padx=8, pady=(8,4))
            self.rec_status = tk.Label(ac, text="Press R to start",
                                       bg=C["card"], fg=C["dim"],
                                       font=tkfont.Font(family="Segoe UI",
                                                        size=9))
            self.rec_status.pack(anchor="w", padx=8)

        _btn(ac, "  NEW LABEL  (N)  ",
             self._ask_label, bg=C["card"]).pack(fill="x", padx=8, pady=2)
        _btn(ac, "  SAVE & QUIT  (Q)  ",
             self._save_quit, bg=C["card"]).pack(fill="x", padx=8,
                                                  pady=(2,8))

        # dataset card
        dc = card("DATASET", 3)
        self.dataset_lbl = tk.Label(dc, text="", bg=C["card"], fg=C["sub"],
                                    font=tkfont.Font(family="Segoe UI",
                                                     size=8),
                                    justify="left", wraplength=250)
        self.dataset_lbl.pack(anchor="w", padx=8, pady=6)
        self._refresh_dataset()

    def _refresh_dataset(self):
        counts = self.clf.label_counts()
        if not counts:
            self.dataset_lbl.configure(text="(empty)")
            return
        total = sum(counts.values())
        labels = sorted(counts.keys())
        self.dataset_lbl.configure(
            text=f"{len(labels)} labels  •  {total} samples\n"
                 + ", ".join(f"{l}:{counts[l]}" for l in labels))

    def _update_progress(self):
        n = self.clf.label_counts().get(self.label, 0) if self.label else 0
        self.count_lbl.configure(text=f"{n} / {self.TARGET}")
        bw = self.prog_canvas.winfo_width()
        fill = min(1.0, n/self.TARGET) * bw
        self.prog_canvas.coords(self.prog_bar, 0, 0, fill, 14)
        col = C["green"] if n >= self.TARGET else C["accent"]
        self.prog_canvas.itemconfigure(self.prog_bar, fill=col)

    def _ask_label(self):
        new = simpledialog.askstring("Label",
                                     "Enter label (letter, word, etc.):",
                                     parent=self.root)
        if new and new.strip():
            self.label = new.strip()
            self.label_lbl.configure(text=self.label[:3])
            self._update_progress()

    def _capture(self):
        if self.mode != "static" or not self.label:
            return
        if self._last_left is None and self._last_right is None:
            self.count_lbl.configure(text="No hand!", fg=C["red"])
            self.root.after(800, lambda: self._update_progress())
            return
        self.clf.add_sample(combine_hands(self._last_left,
                                          self._last_right), self.label)
        self.flash = 1.0
        self._update_progress()
        self._refresh_dataset()

    def _toggle_rec(self):
        if self.mode != "dynamic" or not self.label:
            return
        if self.recording:
            self.recording = False
            self.rec_btn.configure(text="  ◉  RECORD  (R)  ", bg=C["card"])
            if len(self.rec_frames) >= 8:
                seq = build_sequence_features(self.rec_frames)
                self.clf.add_sample(seq, self.label)
                self.rec_status.configure(
                    text=f"✓ Saved! ({len(self.rec_frames)} frames)",
                    fg=C["green"])
                self.flash = 1.0
                self._update_progress()
                self._refresh_dataset()
            else:
                self.rec_status.configure(text="Too short — try again",
                                          fg=C["red"])
            self.rec_frames = []
        else:
            self.recording = True
            self.rec_frames = []
            self.rec_btn.configure(text="  ■  STOP  (R)  ", bg=C["rec"])
            self.rec_status.configure(text="Recording...", fg=C["rec"])

    def _save_quit(self):
        path = STATIC_PATH if self.mode=="static" else DYNAMIC_PATH
        self.clf.save(path)
        messagebox.showinfo("Saved",
            f"Dataset saved!\n{self.clf.label_counts()}", parent=self.root)
        self._quit()

    def _loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._loop)
            return
        frame = cv2.flip(frame, 1)
        left, right, wrist = self.backend.process(frame)
        self.backend.draw(frame)
        self._last_left = left
        self._last_right = right
        self.frame_count += 1

        if self.recording and wrist:
            self.rec_frames.append(
                {"left": left, "right": right, "wrist": wrist})
            if hasattr(self, 'rec_status'):
                self.rec_status.configure(
                    text=f"Recording...  {len(self.rec_frames)} frames")

        h, w = frame.shape[:2]

        # recording border
        if self.recording:
            pulse = int(180 + 75 * math.sin(self.frame_count * 0.18))
            cv2.rectangle(frame, (0,0), (w-1,h-1), (30,30,pulse), 5)

        # flash
        if self.flash > 0:
            ov = np.full_like(frame, 255, dtype=np.uint8)
            cv2.addWeighted(ov, self.flash*0.3, frame,
                            1-self.flash*0.3, 0, frame)
            self.flash = max(0.0, self.flash - 0.12)

        # hint
        hint = "SPACE=capture  N=new label  Q=save+quit" \
               if self.mode=="static" \
               else "R=start/stop  N=new label  Q=save+quit"
        cv2.putText(frame, hint, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((cw,ch), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self.root.after(1, self._loop)

    def _quit(self):
        self.running = False
        try: self.cap.release()
        except: pass
        self.root.destroy()


def run_static():  RecorderApp("static")
def run_dynamic(): RecorderApp("dynamic")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode == "static":   run_static()
    elif mode == "dynamic": run_dynamic()
    else: print(__doc__)
