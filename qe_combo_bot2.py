import time
import threading
import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import pygetwindow as gw
from pynput import keyboard as kb

# ── CONFIG ──────────────────────────────────────────
PRESS_DURATION  = 0.06   # seconds to hold each key
PRESS_DELAY     = 0.13   # gap between presses
SCAN_INTERVAL   = 0.25   # how often to check screen
# ────────────────────────────────────────────────────

bot_active  = False
quit_flag   = False
last_state  = None
pyautogui.PAUSE     = 0
pyautogui.FAILSAFE  = True


# ── HOTKEYS ─────────────────────────────────────────
def on_press(key):
    global bot_active, quit_flag
    try:
        if key == kb.Key.f8:
            bot_active = not bot_active
            print(f"[BOT] {'ON 🟢' if bot_active else 'OFF 🔴'}")
        elif key == kb.Key.f9:
            quit_flag = True
            print("[BOT] Quitting...")
            return False
    except Exception:
        pass

listener = kb.Listener(on_press=on_press)
listener.start()


# ── FOCUS ROBLOX ────────────────────────────────────
def focus_roblox():
    """Click on Roblox window to make sure keypresses go there."""
    try:
        wins = [w for w in gw.getAllWindows()
                if 'roblox' in w.title.lower() and w.width > 100]
        if wins:
            w = wins[0]
            cx = w.left + w.width  // 2
            cy = w.top  + w.height // 2
            pyautogui.click(cx, cy)
            time.sleep(0.1)
            return True
    except Exception as e:
        print(f"[WARN] Could not focus Roblox: {e}")
    return False


# ── SCREEN CAPTURE ───────────────────────────────────
def grab():
    img = ImageGrab.grab()
    return np.array(img)   # RGB


# ── KEY DETECTION ────────────────────────────────────
def find_keys(frame):
    """
    Locate the Q/E key icons along the bottom combo bar.
    Returns list of (x_start, x_end) in full-frame coords.
    """
    h, w = frame.shape[:2]

    # Focus on bottom 17% of screen where the combo bar lives
    y1 = int(h * 0.83)
    y2 = int(h * 0.83) + int(h * 0.07)
    x1 = int(w * 0.25)
    x2 = int(w * 0.76)

    bar = frame[y1:y2, x1:x2]
    bh, bw = bar.shape[:2]
 
    Image.fromarray(bar).save("debug_bar.png")
    
    # Key icons have white backgrounds — find bright columns
    gray = bar.mean(axis=2)
    bright = gray > 230

    col_sums = bright.sum(axis=0)
    threshold = bh * 0.15
    bright_cols = np.where(col_sums > threshold)[0]

    if len(bright_cols) < 10:
        return []

    # Cluster contiguous bright columns into individual keys
    clusters = []
    start = bright_cols[0]
    prev  = bright_cols[0]
    for c in bright_cols[1:]:
        if c - prev > 12:
            clusters.append((start + x1, prev + x1))
            start = c
        prev = c
    clusters.append((start + x1, prev + x1))

    # Filter to square-ish blobs
    valid = []
    for xs, xe in clusters:
        width = xe - xs
        if 25 < width < 120:
            valid.append((xs, xe, y1, y2))

    return valid


def classify_key(frame, xs, xe, y1, y2):
    """
    Returns ('Q' or 'E', is_lit).
    
    Detection logic (derived from pixel analysis of the actual icons):
    - E has a flat top bar → many dark pixels (≥50) in the top 4 rows of the icon
    - Q has a round top    → fewer dark pixels (<50) in the top 4 rows
    - 'lit' = the icon background is bright white (still needs pressing)
    - 'dim' = the icon is greyed out (already pressed)
    """
    key_img = frame[y1:y2, xs:xe]   # RGB
    gray    = key_img.mean(axis=2)

    # Lit/dim: check average brightness of the whole icon
    avg_brightness = gray.mean()
    is_lit = avg_brightness > 160   # lit keys are brighter

    # Q vs E: count dark pixels in top 4 rows of the icon
    top_rows       = gray[:4, :]
    dark_in_top    = (top_rows < 80).sum()

    letter = 'E' if dark_in_top >= 50 else 'Q'

    return letter, is_lit


def detect_combo(frame):
    """Returns list of (letter, is_lit) for all keys in the combo bar."""
    keys = find_keys(frame)
    print("Keys found:", len(keys))
    if not keys:
        return []

    result = []
    for xs, xe, y1, y2 in keys:
        letter, lit = classify_key(frame, xs, xe, y1, y2)
        result.append((letter, lit))
    return result


# ── KEY PRESSER ──────────────────────────────────────
def press_keys(combo):
    """Press only the lit (unfinished) keys."""
    to_press = [k for k, lit in combo if lit]
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


# ── MAIN LOOP ────────────────────────────────────────
def main():
    global last_state, bot_active

    print("=" * 50)
    print("  QE Combo Bot v2")
    print("=" * 50)
    print("  F8  → Toggle ON/OFF")
    print("  F9  → Quit")
    print("  Mouse top-left = emergency stop")
    print("=" * 50)
    print("  Waiting... Switch to Roblox and press F8")
    print()

    focused = False

    while not quit_flag:
        time.sleep(SCAN_INTERVAL)

        if not bot_active:
            focused = False
            continue
            print("Scanning...")
        # Focus Roblox once when we turn on
        if not focused:
            focused = focus_roblox()
            print("Focused:", focused)

        try:
            frame = grab()
        except Exception as e:
            print(f"[ERR] Capture: {e}")
            continue

        combo = detect_combo(frame)

        if len(combo) < 2:
            continue

        full_str = ' '.join(k for k, _ in combo)
        lit_str  = ' '.join(k for k, lit in combo if lit)

        if not lit_str:
            last_state = None
            continue

        # Skip if nothing changed
        state_key = (full_str, lit_str)
        if state_key == last_state:
            continue
        last_state = state_key

        print(f"[BOT] Combo: {full_str}  |  Left: {lit_str}")

        t = threading.Thread(target=press_keys, args=(combo,), daemon=True)
        t.start()

    listener.stop()
    print("[BOT] Done.")


if __name__ == "__main__":
    main()
