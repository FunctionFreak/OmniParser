#!/usr/bin/env python3
"""
process_images.py ‒ OmniParser YOLOv8 detector (no caption model)
===============================================================
Reads screenshots from  **./test/**, draws multicolour boxes like the official
OmniParser demo, and writes the annotated image plus a tab-separated text file
to **./result/**.

This version **matches the Hugging Face Space hyper-parameters**:
    • imgsz   = 1280   (model’s native train size)
    • conf    = 0.05   (low threshold = high recall)
    • iou     = 0.15   (looser NMS)
    • max_det = 3000   (allow thousands of boxes)
    • augment = True   (test-time augmentation for extra recall)

Run inside your activated venv:
    python process_images.py
"""
from __future__ import annotations

from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

# ──────────── CONFIG ────────────
ROOT_DIR   = Path(__file__).resolve().parent
TEST_DIR   = ROOT_DIR / "test"
RESULT_DIR = ROOT_DIR / "result"
MODEL_PATH = ROOT_DIR / "weights" / "icon_detect" / "model.pt"

IMG_SIZE   = 1088
CONF_THRES = 0.09
IOU_THRES  = 0.10
MAX_DET    = 3000
AUGMENT    = True      # test-time augmentation

THICKNESS  = 2          # bbox line width
FONT_SCALE = 0.5
FONT_THICK = 2
PADDING    = 5

# Distinct colour palette (BGR)
COLOURS = [
    (255,  56,  56), (255, 159,  56), (255, 214,  56), ( 56, 255,  56),
    ( 56, 255, 255), ( 56, 159, 255), (180,  56, 255), (255,  56, 180)
]

# ──────────── SET-UP ────────────
RESULT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔄  Loading YOLO detector on {device} …")
model = YOLO(str(MODEL_PATH)).to(device)
font  = cv2.FONT_HERSHEY_SIMPLEX


def contrast_text(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white for good contrast on the given BGR colour."""
    r, g, b = bgr[2], bgr[1], bgr[0]
    return (0, 0, 0) if 0.299*r + 0.587*g + 0.114*b > 160 else (255, 255, 255)


# ──────────── MAIN LOOP ────────────

def main() -> None:
    images = sorted(TEST_DIR.glob("*.[jp][pn]g"))
    if not images:
        raise SystemExit(f"No .jpg or .png images found in {TEST_DIR}")

    for img_path in images:
        print(f"🖼️   Processing {img_path.name}")

        # — run detection —
        res = model(img_path,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    iou=IOU_THRES,
                    max_det=MAX_DET,
                    augment=AUGMENT)[0]
        boxes = res.boxes

        img = cv2.imread(str(img_path))
        if img is None:
            print("   ⚠️  Failed to read image – skipping.")
            continue

        txt_path = RESULT_DIR / f"{img_path.stem}_det.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf, cls = float(b.conf[0]), int(b.cls[0])
                colour = COLOURS[i % len(COLOURS)]

                # rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, THICKNESS)

                # label bar
                label = str(i)
                (tw, th), _ = cv2.getTextSize(label, font, FONT_SCALE, FONT_THICK)
                bar1 = (x1, max(0, y1 - th - 2*PADDING))
                bar2 = (x1 + tw + 2*PADDING, y1)
                cv2.rectangle(img, bar1, bar2, colour, -1)
                cv2.putText(img, label, (bar1[0] + PADDING, bar2[1] - PADDING),
                            font, FONT_SCALE, contrast_text(colour), FONT_THICK, cv2.LINE_AA)

                # log to txt
                f.write(f"{i}\t{x1}\t{y1}\t{x2}\t{y2}\t{conf:.3f}\t{cls}\n")

        out_img = RESULT_DIR / f"{img_path.stem}_det{img_path.suffix}"
        cv2.imwrite(str(out_img), img)
        print(f"   ➜ saved {out_img.name} & {txt_path.name}")

    print("\n✅  All images processed – see result/ folder")


if __name__ == "__main__":
    main()