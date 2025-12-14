import time
import threading
import json
from dataclasses import dataclass

import mss
import numpy as np
import cv2
import pygetwindow as gw
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


@dataclass
class WinInfo:
    title: str
    left: int
    top: int
    width: int
    height: int


def list_windows() -> list[WinInfo]:
    wins = []
    for w in gw.getAllWindows():
        try:
            if not w.title:
                continue
            if w.width <= 0 or w.height <= 0:
                continue
            if w.width < 200 or w.height < 150:
                continue
            wins.append(WinInfo(w.title, w.left, w.top, w.width, w.height))
        except Exception:
            continue

    wins.sort(key=lambda x: (("minecraft" not in x.title.lower()), x.title.lower()))
    return wins


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MB Vision - Window Picker Preview")

        self.windows: list[WinInfo] = []
        self.selected: WinInfo | None = None
        self.running = False
        self.last_frame = None

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        self.combo = ttk.Combobox(top, state="readonly", width=80)
        self.combo.pack(side="left", fill="x", expand=True)

        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Start Preview", command=self.start).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Stop", command=self.stop).pack(side="left", padx=(8, 0))

        ttk.Button(top, text="Announce to MB", command=self.announce).pack(side="left", padx=(8, 0))

        self.preview_label = ttk.Label(root)
        self.preview_label.pack(padx=8, pady=(0, 8))

        self.status = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self.status).pack(fill="x", padx=8, pady=(0, 8))

        self.refresh()
        self.combo.bind("<<ComboboxSelected>>", self.on_select)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def refresh(self):
        self.windows = list_windows()
        titles = [w.title for w in self.windows]
        self.combo["values"] = titles

        if titles:
            self.combo.current(0)
            self.selected = self.windows[0]
            self.status.set(f"Selected: {self.selected.title}")
        else:
            self.selected = None
            self.status.set("No windows found. Open Minecraft (or the app you want) and hit Refresh.")

    def on_select(self, _evt=None):
        idx = self.combo.current()
        if 0 <= idx < len(self.windows):
            self.selected = self.windows[idx]
            self.status.set(f"Selected: {self.selected.title}")

    def start(self):
        if self.running:
            return
        if not self.selected:
            self.status.set("No window selected.")
            return
        self.running = True
        self.status.set("Preview running… (use Stop)")
        t = threading.Thread(target=self.capture_loop, daemon=True)
        t.start()
        self.update_ui_loop()

    def stop(self):
        self.running = False
        self.status.set("Stopped.")

    def announce(self):
        if not self.selected:
            print("[MB_VISION] NOT_READY no_window_selected")
            return

        payload = {
            "ready": True,
            "title": self.selected.title,
            "rect": {
                "left": self.selected.left,
                "top": self.selected.top,
                "width": self.selected.width,
                "height": self.selected.height,
            },
            "ts": time.time(),
        }
        print("[MB_VISION] READY " + json.dumps(payload))

        try:
            with open("mb_vision_status.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print("[MB_VISION] WARN could_not_write_status_file", repr(e))

    def capture_loop(self):
        with mss.mss() as sct:
            while self.running:
                sel = self.selected
                if not sel:
                    time.sleep(0.1)
                    continue

                mon = {"left": sel.left, "top": sel.top, "width": sel.width, "height": sel.height}
                try:
                    img = np.array(sct.grab(mon))  # BGRA
                    frame = img[:, :, :3]          # BGR
                    self.last_frame = frame
                except Exception as e:
                    self.last_frame = None
                    self.status.set(f"Capture error: {e!r}")
                    time.sleep(0.2)

                time.sleep(0.01)

    def update_ui_loop(self):
        if not self.running:
            return

        frame = self.last_frame
        if frame is not None:
            h, w = frame.shape[:2]
            target_w = 520
            scale = target_w / max(w, 1)
            target_h = max(1, int(h * scale))
            small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            tkimg = ImageTk.PhotoImage(im)

            self.preview_label.configure(image=tkimg)
            self.preview_label.image = tkimg

            mean_luma = float(rgb.mean())
            self.status.set(f"Preview: {self.selected.title} | {w}x{h} | mean={mean_luma:.1f}")

        self.root.after(33, self.update_ui_loop)

    def on_close(self):
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
