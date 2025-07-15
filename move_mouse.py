#!/usr/bin/env python3
"""
move_mouse_scaled.py
--------------------
Move the mouse to coordinates taken from a *pixel-resolution* screenshot.
The script automatically converts those pixel coords to the logical *point*
grid that macOS (and thus PyAutoGUI) uses.

Usage:
    python move_mouse_scaled.py <x_pixel> <y_pixel>

Example (coords from a 2940×1912 Retina screenshot):
    python move_mouse_scaled.py 1000 500

Dependencies:
    pip install pyautogui pillow
"""

import sys
import pyautogui
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────
def get_scale_factor() -> float:
    """
    Return the display’s backing-scale factor, e.g. 2.0 on a Retina MacBook Air.
    """
    width_pt, _ = pyautogui.size()            # logical points
    width_px, _ = pyautogui.screenshot().size # physical pixels
    return width_px / width_pt if width_pt else 1.0


def convert_px_to_pt(x_px: int, y_px: int, scale: float) -> tuple[int, int]:
    """
    Convert pixel coordinates → point coordinates, rounding to nearest integer.
    """
    return round(x_px / scale), round(y_px / scale)


def move_mouse(x_px: int, y_px: int) -> bool:
    """
    Convert the supplied pixel-based coordinates and move the cursor.
    Returns True on success, False if the point lies outside the desktop.
    """
    scale = get_scale_factor()
    x_pt, y_pt = convert_px_to_pt(x_px, y_px, scale)

    screen_w_pt, screen_h_pt = pyautogui.size()
    if not (0 <= x_pt < screen_w_pt and 0 <= y_pt < screen_h_pt):
        print(
            f"Error: converted coordinates ({x_pt}, {y_pt}) are outside "
            f"screen bounds {screen_w_pt}×{screen_h_pt} pt"
        )
        return False

    print(f"Backing scale factor detected: {scale:.2f}×")
    print(
        f"Moving mouse (screenshot px → screen pt): "
        f"({x_px}, {y_px}) → ({x_pt}, {y_pt})"
    )

    pyautogui.moveTo(x_pt, y_pt)

    # Show where the pointer ended up
    cur_x, cur_y = pyautogui.position()
    print(f"Mouse now at: ({cur_x}, {cur_y}) pt")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python move_mouse_scaled.py <x_pixel> <y_pixel>")
        w_pt, h_pt = pyautogui.size()
        w_px, h_px = pyautogui.screenshot().size
        print(
            f"Current point size: {w_pt}×{h_pt}  |  "
            f"Current screenshot size: {w_px}×{h_px}"
        )
        sys.exit(1)

    try:
        x_px = int(sys.argv[1])
        y_px = int(sys.argv[2])
    except ValueError:
        print("Error: coordinates must be integers")
        sys.exit(1)

    if not move_mouse(x_px, y_px):
        sys.exit(1)


if __name__ == "__main__":
    main()