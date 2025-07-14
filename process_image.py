#!/usr/bin/env python3
"""
process_images.py
-----------------
Batch-detect UI elements with OmniParser’s **YOLOv8 detector only** (no caption model)
and save nicely-annotated outputs that match the look of Microsoft’s demo images.

**Pipeline**
* Reads every *.jpg* / *.png* in **./test/**
* Runs the detector at higher recall settings (imgsz 1280, conf 0.05, max_det 1000)
* Draws a *colored* bounding-box for each detection, with an index label on a filled
  bar — visually identical to OmniParser’s official screenshots
* Writes two files per image into **./result/**
  1. `<name>_det.jpg/png` ─ the annotated screenshot
  2. `<name>_det.txt`     ─ tab-separated rows: `id  x1 y1 x2 y2  conf  cls`

Run inside your activated venv:
    python process_images.py
"""
from __future__ import annotations

from pathlib import Path
import random
import cv2                  # OpenCV drawing API
import numpy as np
from ultralytics import YOLO
import torch

# ────────────────── CONFIG ──────────────────
ROOT_DIR   = Path(__file__).resolve().parent
TEST_DIR   = ROOT_DIR / "test"
RESULT_DIR = ROOT_DIR / "result"
MODEL_PATH = ROOT_DIR / "weights/icon_detect/model.pt"

IMG_SIZE   = 640      # feed size (multiple of 32)
CONF_THRES = 0.05      # much lower → higher recall
MAX_DET    = 1000      # keep plenty of boxes
THICKNESS  = 2         # bbox line thickness
FONT_SCALE = 0.5       # label text size
FONT_THICK = 2
PADDING    = 5         # px around text inside label bar

# Distinct color palette (BGR tuples)
COLOR_PALETTE = [
    (255, 56, 56),  # red
    (255, 159, 56), # orange
    (255, 214, 56), # yellow
    (56, 255, 56),  # green
    (56, 255, 255), # cyan
    (56, 159, 255), # blue
    (180, 56, 255), # violet
    (255, 56, 180), # pink
]

# ────────────────── SET-UP ──────────────────
RESULT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔄  Loading YOLO detector on {device} …")
model = YOLO(str(MODEL_PATH)).to(device)
font = cv2.FONT_HERSHEY_SIMPLEX

# ────────────────── HELPERS ──────────────────

def get_contrasting_text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white depending on luminance of the given BGR colour."""
    r, g, b = bgr[2], bgr[1], bgr[0]  # convert to RGB order for formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)

# ────────────────── MAIN LOOP ──────────────────

def main() -> None:
    images = sorted(TEST_DIR.glob("*.[jp][pn]g"))
    if not images:
        raise SystemExit(f"No .jpg/.png images found in {TEST_DIR}")

    for img_path in images:
        print(f"🖼️   Processing {img_path.name}")

        # --- run detection
        results = model(img_path,
                        imgsz=IMG_SIZE,
                        conf=CONF_THRES,
                        max_det=MAX_DET)[0]
        boxes = results.boxes

        # load image via OpenCV
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"   ⚠️  Failed to read {img_path}; skipping.")
            continue
        h, w = img_bgr.shape[:2]

        # open txt file to save box data
        txt_path = RESULT_DIR / f"{img_path.stem}_det.txt"
        with txt_path.open("w", encoding="utf-8") as f_out:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])

                color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                text  = str(idx)

                # draw rectangle
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, THICKNESS)

                # compute label background
                (text_w, text_h), _ = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICK)
                bar_x1, bar_y1 = x1, max(0, y1 - text_h - 2 * PADDING)
                bar_x2, bar_y2 = x1 + text_w + 2 * PADDING, y1
                cv2.rectangle(img_bgr, (bar_x1, bar_y1), (bar_x2, bar_y2), color, -1)

                # put label text
                text_color = get_contrasting_text_color(color)
                cv2.putText(img_bgr, text,
                            (bar_x1 + PADDING, bar_y2 - PADDING),
                            font, FONT_SCALE, text_color, FONT_THICK, cv2.LINE_AA)

                # write to txt file
                f_out.write(f"{idx}\t{x1}\t{y1}\t{x2}\t{y2}\t{conf:.3f}\t{cls}\n")

        # save annotated image (keeping extension)
        out_img_path = RESULT_DIR / f"{img_path.stem}_det{img_path.suffix}"
        cv2.imwrite(str(out_img_path), img_bgr)
        print(f"   ➜ saved {out_img_path.name} & {txt_path.name}")

    print("\n✅  All images processed – check the result/ folder!")


if __name__ == "__main__":
    main()