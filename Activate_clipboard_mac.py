import time
import cv2
import numpy as np
import mss
import pytesseract
import pyperclip
import platform
import subprocess
import sys
from pynput import keyboard

# macOS Tesseract path
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Hotkeys
HOTKEY_RESELECT = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('r')}
HOTKEY_QUIT     = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('q')}
HOTKEY_ENABLE   = {keyboard.Key.cmd, keyboard.Key.f9}
HOTKEY_DISABLE  = {keyboard.Key.cmd, keyboard.Key.f10}

pressed = set()
monitor = None
running = True
DEBUG = True
Flag = 1

def check_architecture():
    """Verify Python and Tesseract architectures."""
    print(f"Python architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    try:
        result = subprocess.run(["file", tesseract_path or "/usr/local/bin/tesseract"], 
                              capture_output=True, text=True, timeout=5)
        print(f"Tesseract: {result.stdout.strip()}")
    except Exception as e:
        print(f"Could not check Tesseract: {e}")

def choose_monitor_index():
    """Prompt user to choose a display."""
    try:
        with mss.MSS() as sct:
            monitors = sct.monitors
            count_real = len(monitors) - 1
            if count_real <= 1:
                return 1
            print("\nAvailable displays:")
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

def input_roi_coordinates():
    """Get ROI coordinates from user input instead of GUI."""
    print("\n=== Manual ROI Selection ===")
    print("First, take a screenshot to see coordinate ranges.")
    print("You can use macOS screencapture or just estimate from your screen.")
    
    try:
        left = int(input("Enter LEFT coordinate (x): "))
        top = int(input("Enter TOP coordinate (y): "))
        width = int(input("Enter WIDTH: "))
        height = int(input("Enter HEIGHT: "))
        
        if width <= 0 or height <= 0:
            print("Width and height must be positive.")
            return None
        
        return {"left": left, "top": top, "width": width, "height": height}
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return None

def select_roi():
    """Get ROI from user input - no GUI to avoid crashes."""
    try:
        idx = choose_monitor_index()
        with mss.MSS() as sct:
            base = sct.monitors[idx]
            print(f"\nDisplay {idx} bounds: left={base['left']}, top={base['top']}, "
                  f"width={base['width']}, height={base['height']}")
        
        m = input_roi_coordinates()
        if not m:
            return None
        
        return m
    except Exception as e:
        print(f"Error selecting ROI: {e}")
        return None

def preprocess(frame_bgra):
    """Minimal preprocessing - avoid heavy OpenCV operations."""
    try:
        if frame_bgra is None or frame_bgra.size == 0:
            return None
        
        # Minimal processing to avoid crashes
        if len(frame_bgra.shape) == 3 and frame_bgra.shape[2] == 4:
            # BGRA to BGR
            frame_bgr = frame_bgra[:, :, :3]
        else:
            frame_bgr = frame_bgra
        
        # Convert to grayscale
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
        
        # Simple threshold (avoid OTSU which can be slow)
        _, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
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
    """Run OCR with error handling."""
    if img is None:
        return ""
    try:
        config = "--oem 3 --psm 6 -l eng"
        result = pytesseract.image_to_string(img, config=config, timeout=3)
        return result if result else ""
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract not found. Install: brew install tesseract")
        return ""
    except Exception as e:
        return ""

def reselect():
    """Reselect ROI."""
    global monitor
    try:
        m = select_roi()
        if not m:
            if monitor:
                print("Selection canceled. Keeping previous region.")
                return True
            else:
                print("No region selected.")
                return False
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
            print("\n[Hotkey] Reselect region...")
            reselect()
        
        if HOTKEY_QUIT.issubset(pressed):
            print("\n[Hotkey] Quit requested.")
            running = False
        
        if HOTKEY_ENABLE.issubset(pressed) and Flag == 0:
            Flag = 1
            print("[Hotkey] Clipboard copying ENABLED")
        
        if HOTKEY_DISABLE.issubset(pressed) and Flag == 1:
            Flag = 0
            print("[Hotkey] Clipboard copying DISABLED")
    except Exception as e:
        pass

def on_release(key):
    """Handle key release."""
    try:
        pressed.discard(key)
    except:
        pass

def main():
    """Main OCR loop."""
    global running, Flag
    
    print("\n=== Clipboard OCR Launcher ===")
    check_architecture()
    print("\nNote: Using manual coordinate input instead of GUI to avoid crashes.\n")
    
    if not reselect():
        print("No region selected. Exiting.")
        return

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    last_copied = ""
    frame_count = 0
    consecutive_errors = 0
    error_limit = 20

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
                    try:
                        frame = np.array(sct.grab(monitor), dtype=np.uint8)
                    except Exception as e:
                        print(f"Screenshot error: {e}. Reselect region.")
                        if not reselect():
                            break
                        continue
                    
                    if frame is None or frame.size == 0:
                        consecutive_errors += 1
                        time.sleep(0.5)
                        continue
                    
                    # Process image
                    proc = preprocess(frame)
                    if proc is None:
                        consecutive_errors += 1
                        if consecutive_errors > error_limit:
                            print("Too many errors. Reselecting region...")
                            reselect()
                            consecutive_errors = 0
                        time.sleep(0.5)
                        continue
                    
                    # Run OCR
                    text = normalize_text(ocr_image(proc))
                    consecutive_errors = 0
                    
                    frame_count += 1
                    
                    # Debug output
                    if DEBUG and frame_count % 20 == 0:
                        status = f"Text: '{text}'" if text else "No text"
                        print(f"[Frame {frame_count}] {status} | Flag={Flag}")
                    
                    # Copy to clipboard
                    if Flag == 1:
                        if text and text != last_copied:
                            try:
                                pyperclip.copy(text)
                                last_copied = text
                                print(f"[COPIED] {text}")
                            except Exception as e:
                                print(f"Clipboard error: {e}")
                    
                    time.sleep(0.2)
                    
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        print(f"Error: {type(e).__name__}: {e}")
                    if consecutive_errors > error_limit:
                        print(f"Too many errors ({consecutive_errors}). Stopping.")
                        break
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        listener.stop()
        print("Stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()