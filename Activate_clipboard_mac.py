import time
import cv2
import numpy as np
import mss
import pytesseract
import pyperclip
from PIL import Image
import platform
import subprocess
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
    """Verify Python and dependencies."""
    print(f"Python architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    try:
        result = subprocess.run(["file", tesseract_path or "/usr/local/bin/tesseract"], 
                              capture_output=True, text=True, timeout=5)
        print(f"Tesseract: {result.stdout.strip()}")
    except:
        pass

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
        print(f"Error: {e}")
        return 1

def select_roi():
    """Use cv2.selectROI to let user draw rectangle."""
    try:
        with mss.MSS() as sct:
            idx = choose_monitor_index()
            base = sct.monitors[idx]
            
            # Grab screenshot
            scr = sct.grab(base)
            img = np.array(scr)[:, :, :3]  # BGRA to BGR
            
            print("\nDraw a rectangle around the text area you want to capture.")
            print("Press ENTER to confirm, ESC to cancel")
            
            # Use cv2.selectROI for region selection
            r = cv2.selectROI("Select region", img, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            
            x, y, w, h = map(int, r)
            if w == 0 or h == 0:
                return None
            
            # Convert to absolute coordinates
            return {"left": base["left"] + x, "top": base["top"] + y, "width": w, "height": h}
    except Exception as e:
        print(f"Error selecting ROI: {e}")
        return None

def preprocess_image(frame_bgra):
    """
    Process image using PIL (more stable than cv2).
    Convert to grayscale and apply threshold.
    """
    try:
        if frame_bgra is None or frame_bgra.size == 0:
            return None
        
        # Convert numpy array to PIL Image
        if len(frame_bgra.shape) == 3 and frame_bgra.shape[2] == 4:
            # BGRA to RGB
            img_rgb = frame_bgra[:, :, ::-1][:, :, 1:]  # Remove alpha, reverse BGR
            pil_img = Image.fromarray(frame_bgra[:, :, :3], mode='RGB')
        else:
            pil_img = Image.fromarray(frame_bgra)
        
        # Convert to grayscale
        pil_gray = pil_img.convert('L')
        
        # Apply threshold
        pil_bw = pil_gray.point(lambda x: 255 if x > 127 else 0, '1')
        
        # Convert back to numpy for pytesseract
        img_array = np.array(pil_bw)
        return img_array
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
    """Reselect ROI using cv2.selectROI."""
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
    except:
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
                    frame = np.array(sct.grab(monitor), dtype=np.uint8)
                    
                    if frame is None or frame.size == 0:
                        consecutive_errors += 1
                        time.sleep(0.5)
                        continue
                    
                    # Process image using PIL (stable)
                    proc = preprocess_image(frame)
                    if proc is None:
                        consecutive_errors += 1
                        if consecutive_errors > 15:
                            print("Too many errors. Reselecting region...")
                            reselect()
                            consecutive_errors = 0
                        time.sleep(0.5)
                        continue
                    
                    # Run OCR
                    text = normalize_text(ocr_image(proc))
                    consecutive_errors = 0
                    
                    frame_count += 1
                    
                    # Debug output every 15 frames
                    if DEBUG and frame_count % 15 == 0:
                        status = f"'{text}'" if text else "No text"
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
                    
                except mss.exception.ScreenShotError:
                    print("Display changed. Please reselect.")
                    if not reselect():
                        break
                    time.sleep(1)
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 2:
                        print(f"Processing error: {type(e).__name__}")
                    if consecutive_errors > 15:
                        print("Too many errors. Stopping.")
                        break
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        listener.stop()
        print("Exiting.")

if __name__ == "__main__":
    main()