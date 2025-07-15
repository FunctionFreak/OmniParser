#!/usr/bin/env python3
"""
quick_dims.py
-------------
Prints the pixel dimensions of a full-screen screenshot
and the logical screen size that mouse/GUI APIs use.

Requires:  pip install pyautogui pillow
"""

import pyautogui

def main():
    # logical (point) resolution used by the OS event system
    width_pt, height_pt = pyautogui.size()

    # full-screen screenshot (returns a PIL.Image)
    screenshot = pyautogui.screenshot()
    width_px, height_px = screenshot.size

    print(f"screenshot dimension: {width_px} x {height_px}")
    print(f"screen dimension:    {width_pt} x {height_pt}")

if __name__ == "__main__":
    main()