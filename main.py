#!/usr/bin/env python3
"""
main.py  ·  simplified wrapper around the OmniParser YOLO detector
=================================================================
Reads every .jpg / .png in ./test/, draws multi-colour boxes, and writes:

1.  <name>_det.jpg/png  – annotated image
2.  <name>_det.txt      – one line per box formatted as:
        icon[<index>], coordinate [x: <x1>, y: <y1>]

The hyper-parameters match the public Hugging Face Space:
    imgsz = 1280, conf = 0.05, iou = 0.15, max_det = 3000, augment = True

Run with your virtual-env active:
    python main.py
"""
from __future__ import annotations

from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

# ——— configuration ———————————————————————————————————————
ROOT       = Path(__file__).resolve().parent
SRC_DIR    = ROOT / "test"
OUT_DIR    = ROOT / "result"
MODEL_FILE = ROOT / "weights" / "icon_detect" / "model.pt"

IMG_SIZE   = 1280
CONF       = 0.1
IOU        = 0.15
MAX_DET    = 3000
AUGMENT    = True

LINE_THICK = 2
FONT_SCALE = 0.5
FONT_THICK = 2
PADDING    = 5

COLOURS = [  # BGR tuples
    (255,  56,  56), (255, 159,  56), (255, 214,  56), ( 56, 255,  56),
    ( 56, 255, 255), ( 56, 159, 255), (180,  56, 255), (255,  56, 180),
]

# ——— helpers ————————————————————————————————————————

def text_colour(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = bgr[2], bgr[1], bgr[0]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)

# ——— setup —————————————————————————————————————————
OUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔄  Loading YOLO detector on {device} …")
model = YOLO(str(MODEL_FILE)).to(device)
font  = cv2.FONT_HERSHEY_SIMPLEX

# ——— main loop ——————————————————————————————————————
images = sorted(SRC_DIR.glob("*.[jp][pn]g"))
if not images:
    raise SystemExit(f"No images found in {SRC_DIR}")

for img_path in images:
    print(f"🖼️   Processing {img_path.name}")

    res = model(img_path, imgsz=IMG_SIZE, conf=CONF, iou=IOU,
                max_det=MAX_DET, augment=AUGMENT)[0]
    boxes = res.boxes

    img = cv2.imread(str(img_path))
    if img is None:
        print("   ⚠️  Could not read image; skipping.")
        continue

    txt_path = OUT_DIR / f"{img_path.stem}_det.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            colour = COLOURS[i % len(COLOURS)]

            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, LINE_THICK)

            # label bar with index number
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, font, FONT_SCALE, FONT_THICK)
            bar_tl = (x1, max(0, y1 - th - 2 * PADDING))
            bar_br = (x1 + tw + 2 * PADDING, y1)
            cv2.rectangle(img, bar_tl, bar_br, colour, -1)
            cv2.putText(img, label, (bar_tl[0] + PADDING, bar_br[1] - PADDING),
                        font, FONT_SCALE, text_colour(colour), FONT_THICK, cv2.LINE_AA)

            # write formatted line to txt
            f.write(f"icon[{i}], coordinate [x: {x1}, y: {y1}]\n")

    # save annotated image
    out_img = OUT_DIR / f"{img_path.stem}_det{img_path.suffix}"
    cv2.imwrite(str(out_img), img)
    print(f"   ➜ saved {out_img.name} & {txt_path.name}")

print("\n✅  Finished – check the result/ folder!")