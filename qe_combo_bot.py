"""
QE Combo Bot for Roblox minigame
----------------------------------
How it works:
  1. Takes a screenshot of your screen
  2. Scans the combo bar (bottom-centre of screen) for Q and E key icons
  3. Reads which keys are lit up (still to press) vs greyed out (already done)
  4. Auto-presses them in order with configurable speed

Controls (while script is running):
  F8  → Toggle bot ON / OFF
  F9  → Quit

Setup:
  - Run this script BEFORE or DURING the minigame
  - Switch to Roblox window; the bot watches for the combo bar to appear
  - Adjust SCAN_REGION below if your combo bar is in a different spot
"""

import time
import threading
import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
from pynput import keyboard as kb

# ──────────────────────────────────────────────
#  CONFIG  (tweak these to match your screen)
# ──────────────────────────────────────────────

# Region to scan for the combo bar: (left, top, right, bottom) in pixels
# None = full screen (slower but works on any resolution)
# Example for 1920x1080: SCAN_REGION = (500, 900, 1420, 980)
SCAN_REGION = None          # Auto-detect mode

# How long (seconds) to hold each key press
PRESS_DURATION = 0.05

# Gap between key presses (seconds)
PRESS_DELAY = 0.12

# How often to scan the screen for a new combo (seconds)
SCAN_INTERVAL = 0.3

# Brightness threshold: pixels brighter than this = "lit up" (active key)
# Q/E icons have a bright white background when active, dark when done
BRIGHTNESS_THRESHOLD = 180

# Minimum white-pixel ratio inside a key region to count as "lit"
LIT_RATIO_MIN = 0.25

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────
bot_active = False
quit_flag  = False
last_combo = []

pyautogui.PAUSE = 0        # We control timing ourselves
pyautogui.FAILSAFE = True  # Move mouse to top-left to emergency-stop


# ──────────────────────────────────────────────
#  HOTKEY LISTENER
# ──────────────────────────────────────────────
def on_press(key):
    global bot_active, quit_flag
    try:
        if key == kb.Key.f8:
            bot_active = not bot_active
            status = "ON 🟢" if bot_active else "OFF 🔴"
            print(f"\n[BOT] Toggled {status}")
        elif key == kb.Key.f9:
            print("\n[BOT] Quitting...")
            quit_flag = True
            return False
    except Exception:
        pass

listener = kb.Listener(on_press=on_press)
listener.start()


# ──────────────────────────────────────────────
#  SCREEN CAPTURE
# ──────────────────────────────────────────────
def grab_screen():
    """Grab screen (or sub-region) and return as numpy BGR array."""
    if SCAN_REGION:
        img = ImageGrab.grab(bbox=SCAN_REGION)
    else:
        img = ImageGrab.grab()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────
#  COMBO BAR DETECTION
# ──────────────────────────────────────────────

def find_combo_bar(frame):
    """
    Locate the combo bar in the frame.
    The bar is a row of rounded-square key icons near the bottom-centre.
    We look for clusters of white/light squares in a horizontal band.
    Returns (x, y, w, h) of the detected bar region, or None.
    """
    h, w = frame.shape[:2]

    # Focus on the bottom 25% of the screen
    roi_y_start = int(h * 0.72)
    roi_y_end   = int(h * 0.95)
    roi = frame[roi_y_start:roi_y_end, :]

    # Convert to grayscale and threshold for bright white key icons
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours of bright blobs
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect square-ish blobs that look like key icons
    key_blobs = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        area = bw * bh
        # Key icons are roughly square, 40-120px wide on typical screens
        if 0.7 < aspect < 1.4 and 1200 < area < 20000:
            key_blobs.append((x, y + roi_y_start, bw, bh))

    if len(key_blobs) < 2:
        return None, []

    # Sort left to right
    key_blobs.sort(key=lambda b: b[0])
    return key_blobs


def classify_key(frame, blob):
    """
    Given a blob (x,y,w,h), determine if it shows Q or E,
    and whether it is LIT (still needs pressing) or DONE (already pressed).
    Returns ('Q'|'E'|None, lit: bool)
    """
    x, y, w, h = blob
    # Add a small margin inward to avoid border artifacts
    margin = max(2, w // 8)
    roi = frame[y+margin : y+h-margin, x+margin : x+w-margin]

    if roi.size == 0:
        return None, False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Check brightness: a "lit" key icon has a bright white bg
    lit = float(np.mean(gray)) > BRIGHTNESS_THRESHOLD

    # Read the letter by checking pixel density in left vs right half
    # Q has a tail bottom-right; E has horizontal stripes
    # Simpler: look for the template letter region
    # Since the icons are styled consistently, we use a heuristic:
    #   - Q icon has an oval shape with a small tail → more pixels in the bottom-right
    #   - E icon has three horizontal bars → more pixels spread horizontally
    # We'll use a pre-trained miniature template match approach:
    # Actually the simplest reliable method: look at the icon label area.
    # The letter is rendered in dark colour on white (lit) or white on dark (dim).

    # Invert if dark (dim key) so letter pixels are always dark-on-bright
    if not lit:
        gray = cv2.bitwise_not(gray)

    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    h2, w2 = bw.shape
    if h2 == 0 or w2 == 0:
        return None, lit

    # E has more horizontal runs in the middle rows; Q is more circular
    # Use vertical column density difference: E has strong left-column, Q is central
    left_col  = bw[:, :w2//3]
    right_col = bw[:, 2*w2//3:]
    mid_row   = bw[h2//3 : 2*h2//3, :]

    left_density  = np.mean(left_col)
    right_density = np.mean(right_col)
    mid_density   = np.mean(mid_row)

    # E: dense left edge and 3 horizontal bars → high mid_density, high left
    # Q: roughly symmetric circle
    if left_density > right_density * 1.3 and mid_density > 30:
        letter = 'E'
    else:
        letter = 'Q'

    return letter, lit


def detect_combo(frame):
    """
    Detect the current combo sequence from the frame.
    Returns list of ('Q'|'E', lit) tuples.
    """
    blobs = find_combo_bar(frame)
    if not blobs or blobs[0] is None:
        return []

    combo = []
    for blob in blobs:
        letter, lit = classify_key(frame, blob)
        if letter:
            combo.append((letter, lit))

    return combo


# ──────────────────────────────────────────────
#  FALLBACK: OCR-STYLE PIXEL SCAN
# ──────────────────────────────────────────────

def detect_combo_simple(frame):
    """
    Simpler fallback detector that scans a horizontal strip for Q/E icons
    by looking at evenly spaced positions along the bottom combo bar.
    Works well when the bar has a fixed position and spacing.
    """
    h, w = frame.shape[:2]

    # The combo bar in your screenshot spans roughly x: 30%-75% of screen width,
    # y: ~85-92% of screen height
    bar_y1 = int(h * 0.83)
    bar_y2 = int(h * 0.95)
    bar_x1 = int(w * 0.28)
    bar_x2 = int(w * 0.73)

    bar_roi = frame[bar_y1:bar_y2, bar_x1:bar_x2]
    bar_h, bar_w = bar_roi.shape[:2]

    if bar_w < 10 or bar_h < 10:
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)

    # Detect white blobs (key icon backgrounds)
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    key_blobs = []
    for cnt in contours:
        x, y, bw2, bh2 = cv2.boundingRect(cnt)
        area = bw2 * bh2
        aspect = bw2 / bh2 if bh2 > 0 else 0
        if 0.6 < aspect < 1.6 and 800 < area < 30000:
            # Translate back to full-frame coords
            key_blobs.append((x + bar_x1, y + bar_y1, bw2, bh2))

    key_blobs.sort(key=lambda b: b[0])

    # Merge overlapping blobs
    merged = []
    for blob in key_blobs:
        if merged and blob[0] < merged[-1][0] + merged[-1][2] - 5:
            continue
        merged.append(blob)

    combo = []
    for blob in merged:
        letter, lit = classify_key(frame, blob)
        if letter:
            combo.append((letter, lit))

    return combo


# ──────────────────────────────────────────────
#  KEY PRESSER
# ──────────────────────────────────────────────

def press_sequence(sequence):
    """Press only the LIT (unfinished) keys in the sequence."""
    to_press = [key for key, lit in sequence if lit]

    if not to_press:
        return

    print(f"[BOT] Pressing: {' → '.join(to_press)}")

    for key in to_press:
        if not bot_active:
            break
        pyautogui.keyDown(key.lower())
        time.sleep(PRESS_DURATION)
        pyautogui.keyUp(key.lower())
        time.sleep(PRESS_DELAY)


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    global last_combo, bot_active

    print("=" * 50)
    print("  QE Combo Bot")
    print("=" * 50)
    print("  F8  → Toggle ON/OFF")
    print("  F9  → Quit")
    print("  Move mouse to TOP-LEFT corner = emergency stop")
    print("=" * 50)
    print("  Waiting... (press F8 in Roblox window to start)")
    print()

    while not quit_flag:
        time.sleep(SCAN_INTERVAL)

        if not bot_active:
            continue

        try:
            frame = grab_screen()
        except Exception as e:
            print(f"[ERR] Screen capture failed: {e}")
            continue

        # Try primary detector, fall back to simple one
        combo = detect_combo(frame)
        if len(combo) < 2:
            combo = detect_combo_simple(frame)

        if not combo:
            continue

        # Only act if the combo changed (new round started)
        combo_keys = [k for k, _ in combo]
        lit_keys   = [k for k, lit in combo if lit]

        if not lit_keys:
            # All done — wait for next round
            last_combo = []
            continue

        if combo_keys == [k for k, _ in last_combo] and lit_keys == [k for k, lit in last_combo if lit]:
            # Same state as before, don't re-press
            continue

        last_combo = combo
        full_seq   = ' '.join(combo_keys)
        remaining  = ' '.join(lit_keys)
        print(f"[BOT] Full combo: {full_seq}  |  Remaining: {remaining}")

        # Press in a separate thread so we can still react to F8/F9
        t = threading.Thread(target=press_sequence, args=(combo,), daemon=True)
        t.start()

    print("[BOT] Bye!")
    listener.stop()


if __name__ == "__main__":
    main()
