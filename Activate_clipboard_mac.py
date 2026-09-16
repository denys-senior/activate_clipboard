import time
import cv2
import numpy as np
import mss
import pytesseract
import pyperclip
import platform
import subprocess
from pynput import keyboard

# macOS Tesseract path (installed via Homebrew)
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Check Python and Tesseract architecture
def check_architecture():
    """Verify Python and Tesseract are compatible architectures."""
    print(f"Python architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    try:
        result = subprocess.run(["file", tesseract_path or "/usr/local/bin/tesseract"], 
                              capture_output=True, text=True, timeout=5)
        print(f"Tesseract: {result.stdout.strip()}")
    except Exception as e:
        print(f"Could not check Tesseract: {e}")

# macOS hotkeys use Cmd instead of Ctrl
HOTKEY_RESELECT = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('r')}
HOTKEY_QUIT     = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('q')}
HOTKEY_ENABLE   = {keyboard.Key.cmd, keyboard.Key.f9}
HOTKEY_DISABLE  = {keyboard.Key.cmd, keyboard.Key.f10}

pressed = set()
monitor = None
running = True
DEBUG = True
Flag = 1

def choose_monitor_index():
    """Prompt user to choose a display when multiple monitors exist."""
    try:
        with mss.MSS() as sct:
            monitors = sct.monitors
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
    except Exception as e:
        print(f"Error selecting monitor: {e}. Using display 1.")
        return 1

def select_roi():
    """Let user choose a display, then drag a rectangle."""
    try:
        with mss.MSS() as sct:
            idx = choose_monitor_index()
            base = sct.monitors[idx]
            scr = sct.grab(base)
            img = np.array(scr)[:, :, :3]
            r = cv2.selectROI("Select region and press ENTER (ESC to cancel)", img, 
                            showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            x, y, w, h = map(int, r)
            if w == 0 or h == 0:
                return None
            return {"left": base["left"] + x, "top": base["top"] + y, "width": w, "height": h}
    except Exception as e:
        print(f"Error selecting ROI: {e}")
        return None

def preprocess(frame_bgra):
    """Preprocess image for OCR with error handling."""
    try:
        if frame_bgra.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        else:
            frame_bgr = frame_bgra
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if DEBUG:
            try:
                cv2.imwrite("debug_preprocessed.png", thr)
            except:
                pass
        return thr
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def normalize_text(t):
    """Normalize OCR text."""
    try:
        t = t.replace('\x0c', ' ').strip()
        return ' '.join(t.split())
    except:
        return ""

def ocr_image(img):
    """Run OCR with timeout and error handling."""
    if img is None:
        return ""
    try:
        config = "--oem 3 --psm 6 -l eng"
        # Use timeout to prevent hanging
        result = pytesseract.image_to_string(img, config=config, timeout=5)
        return result if result else ""
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract not found. Install with: brew install tesseract")
        return ""
    except Exception as e:
        # Silently skip OCR errors to avoid crashes
        return ""

def reselect():
    """Reselect ROI with error handling."""
    global monitor
    try:
        m = select_roi()
        if not m:
            print("Selection canceled. Keeping previous region." if monitor else "No region selected.")
            return bool(monitor)
        monitor = m
        print(f"Region selected: {monitor}")
        return True
    except Exception as e:
        print(f"Error during reselection: {e}")
        return bool(monitor)

def on_press(key):
    """Handle key press."""
    global Flag, running
    try:
        pressed.add(key)
        
        if HOTKEY_RESELECT.issubset(pressed):
            print("[Hotkey] Reselect region...")
            reselect()
        
        if HOTKEY_QUIT.issubset(pressed):
            print("[Hotkey] Quit requested.")
            running = False
        
        if HOTKEY_ENABLE.issubset(pressed) and Flag == 0:
            Flag = 1
            print("[Hotkey] Clipboard copying ENABLED")
        
        if HOTKEY_DISABLE.issubset(pressed) and Flag == 1:
            Flag = 0
            print("[Hotkey] Clipboard copying DISABLED")
    except Exception as e:
        print(f"Key handler error: {e}")

def on_release(key):
    """Handle key release."""
    try:
        pressed.discard(key)
    except:
        pass

def main():
    """Main OCR loop with comprehensive error handling."""
    global running, Flag
    
    print("\n=== Clipboard OCR Launcher ===")
    check_architecture()
    print("\nDraw a rectangle around the text area you want to capture.")
    
    if not reselect():
        print("No region selected. Exiting.")
        return

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    last_copied = ""
    frame_count = 0
    consecutive_errors = 0

    print("\nOCR running. Hotkeys (macOS):")
    print("  Cmd+Shift+R - Reselect region")
    print("  Cmd+Shift+Q - Quit")
    print("  Cmd+F9      - Enable clipboard copying")
    print("  Cmd+F10     - Disable clipboard copying")
    print(f"Clipboard copying is currently: {'ENABLED' if Flag == 1 else 'DISABLED'}\n")

    try:
        with mss.MSS() as sct:
            while running:
                try:
                    # Grab screenshot
                    frame = np.array(sct.grab(monitor))
                    
                    # Process image
                    proc = preprocess(frame)
                    if proc is None:
                        consecutive_errors += 1
                        if consecutive_errors > 10:
                            print("Too many preprocessing errors. Reselect region.")
                            reselect()
                            consecutive_errors = 0
                        time.sleep(0.5)
                        continue
                    
                    # Run OCR
                    text = normalize_text(ocr_image(proc))
                    consecutive_errors = 0  # Reset error counter on success
                    
                    frame_count += 1
                    
                    # Debug output
                    if DEBUG and frame_count % 15 == 0:
                        if text:
                            print(f"[Frame {frame_count}] Text: '{text}' | Flag={Flag}")
                        else:
                            print(f"[Frame {frame_count}] No text detected | Flag={Flag}")
                    
                    # Copy to clipboard
                    if Flag == 1:
                        if text and text != last_copied:
                            try:
                                pyperclip.copy(text)
                                last_copied = text
                                print(f"[COPIED] {text}")
                            except Exception as e:
                                print(f"Clipboard error: {e}")
                    
                    time.sleep(0.15)  # ~6-7 fps, more stable
                    
                except mss.exception.ScreenShotError:
                    print("Display changed. Reselecting region...")
                    if not reselect():
                        break
                    time.sleep(1)
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        print(f"Processing error (attempt {consecutive_errors}): {type(e).__name__}")
                    if consecutive_errors > 5:
                        print("Multiple errors detected. Consider reselecting region.")
                        consecutive_errors = 0
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected.")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        listener.stop()
        print("Stopped.")

if __name__ == "__main__":
    main()