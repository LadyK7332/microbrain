import time
import numpy as np
import cv2
import mss
import pygetwindow as gw

WINDOW_TITLE_CONTAINS = "Minecraft"  # change if needed


def find_window():
    wins = [w for w in gw.getAllWindows() if WINDOW_TITLE_CONTAINS.lower() in w.title.lower()]
    if not wins:
        return None
    return wins[0]


def main():
    w = find_window()
    if w is None:
        print("Could not find a window containing:", WINDOW_TITLE_CONTAINS)
        print("Open Minecraft, then run again.")
        return

    left, top, width, height = w.left, w.top, w.width, w.height
    print("Capturing:", w.title)
    print("Rect:", left, top, width, height)

    mon = {"left": left, "top": top, "width": width, "height": height}

    with mss.mss() as sct:
        last = time.time()
        frames = 0
        while True:
            img = np.array(sct.grab(mon))  # BGRA
            frame = img[:, :, :3]          # BGR
            frames += 1

            now = time.time()
            if now - last >= 1.0:
                print("FPS:", frames)
                frames = 0
                last = now

            cv2.imshow("MC Capture", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
