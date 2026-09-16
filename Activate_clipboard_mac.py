import time
import cv2
import numpy as np
import mss
import pytesseract
import pyperclip
from pynput import keyboard

# macOS Tesseract path (installed via Homebrew)
# Try common Homebrew locations
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback to typical Homebrew install locations
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# macOS hotkeys use Cmd instead of Ctrl
HOTKEY_RESELECT = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('r')}
HOTKEY_QUIT     = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('q')}
HOTKEY_ENABLE   = {keyboard.Key.cmd, keyboard.Key.f9}  # Enable clipboard copying
HOTKEY_DISABLE  = {keyboard.Key.cmd, keyboard.Key.f10} # Disable clipboard copying

pressed = set()
monitor = None
running = True
DEBUG = True  # Enable debug output
Flag = 1  # Start with clipboard copying enabled

def choose_monitor_index():
    """Prompt user to choose a display when multiple monitors exist; return index in sct.monitors."""
    with mss.mss() as sct:
        monitors = sct.monitors  # [0] is virtual bounding box; [1..N] are real displays
        count_real = len(monitors) - 1
        if count_real <= 1:
            return 1
        print("Available displays:")
        for i in range(1, len(monitors)):
            mon = monitors[i]
            print(f"  {i}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']})")
    while True:
        sel = input(f"Select display [1-{count_real}] (Enter=1): ").strip()
        if sel == "":
            return 1
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= count_real:
                return idx
        print("Invalid selection. Try again.")

def select_roi():
    """Let user choose a display, then drag a rectangle; return absolute mss monitor dict."""
    with mss.mss() as sct:
        idx = choose_monitor_index()
        base = sct.monitors[idx]
        scr = sct.grab(base)
        img = np.array(scr)[:, :, :3]  # BGRA -> BGR
        # Show selection window
        r = cv2.selectROI("Select region and press ENTER (ESC to cancel)", img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select region and press ENTER (ESC to cancel)")
        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            return None
        # Convert to absolute screen coordinates
        return {"left": base["left"] + x, "top": base["top"] + y, "width": w, "height": h}

def preprocess(frame_bgra):
    """Basic preprocessing to improve OCR: grayscale + Otsu threshold."""
    # frame_bgra may include alpha channel
    if frame_bgra.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    else:
        frame_bgr = frame_bgra
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Light blur helps suppress noise; adaptive threshold via Otsu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save preprocessed image for debugging
    if DEBUG:
        cv2.imwrite("debug_preprocessed.png", thr)
    
    return thr

def normalize_text(t):
    # Collapse whitespace and strip common noise
    t = t.replace('\x0c', ' ').strip()
    return ' '.join(t.split())

def ocr_image(img):
    # Tesseract config tuned for a block of text (psm 6). Adjust if your text is one line (psm 7).
    config = "--oem 3 --psm 6 -l eng"
    return pytesseract.image_to_string(img, config=config)

def reselect():
    global monitor
    m = select_roi()
    if not m:
        print("Selection canceled. Keeping previous region." if monitor else "No region selected; exiting.")
        return bool(monitor)
    monitor = m
    print(f"Region selected: {monitor}")
    return True

def on_press(key):
    # Track pressed keys and handle hotkeys
    global Flag
    pressed.add(key)
    
    if HOTKEY_RESELECT.issubset(pressed):
        print("[Hotkey] Reselect region...")
        reselect()
    
    if HOTKEY_QUIT.issubset(pressed):
        print("[Hotkey] Quit requested.")
        global running
        running = False
    
    if HOTKEY_ENABLE.issubset(pressed) and Flag == 0:
        Flag = 1
        print("[Hotkey] Clipboard copying ENABLED (Flag = 1)")
    
    if HOTKEY_DISABLE.issubset(pressed) and Flag == 1:
        Flag = 0
        print("[Hotkey] Clipboard copying DISABLED (Flag = 0)")

def on_release(key):
    pressed.discard(key)

def main():
    global running, Flag
    print("Draw a rectangle around the text area you want to capture.")
    if not reselect():
        return

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    last_copied = ""
    frame_count = 0

    with mss.mss() as sct:
        print("OCR running. Hotkeys (macOS):")
        print("  Cmd+Shift+R - Reselect region")
        print("  Cmd+Shift+Q - Quit")
        print("  Cmd+F9      - Enable clipboard copying")
        print("  Cmd+F10     - Disable clipboard copying")
        print(f"Clipboard copying is currently: {'ENABLED' if Flag == 1 else 'DISABLED'}")
        print("DEBUG MODE: Showing all detected text")
        
        while running:
            try:
                frame = np.array(sct.grab(monitor))
                proc = preprocess(frame)
                text = normalize_text(ocr_image(proc))
                
                frame_count += 1
                
                # Debug output
                if DEBUG and frame_count % 10 == 0:  # Every 10 frames
                    print(f"[Frame {frame_count}] Detected text: '{text}' | Flag={Flag}")
                
                # Copy only if Flag is 1
                if Flag == 1:
                    if text and text != last_copied:
                        pyperclip.copy(text)
                        last_copied = text
                        print(f"[COPIED TO CLIPBOARD] {text}")
                    elif not text and frame_count % 30 == 0:
                        print("[No text detected in region]")
                else:
                    # When Flag is 0, still detect but don't copy
                    if text and frame_count % 30 == 0:
                        print(f"[Detected but NOT copied (Flag=0)] {text}")

                # Modest frame rate prevents CPU burn
                time.sleep(0.1)  # ~10 fps
                
            except mss.exception.ScreenShotError:
                # If monitor coordinates are invalid (display changes), try reselect
                print("Screen changed. Please reselect region (Cmd+Shift+R).")
                time.sleep(0.5)
            except KeyboardInterrupt:
                running = False

    listener.stop()
    print("Stopped.")

if __name__ == "__main__":
    main()