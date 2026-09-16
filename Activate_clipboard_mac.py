#!/usr/bin/env python3
import os
import sys

# Disable CPU optimizations
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/lib'

import time
import cv2
import numpy as np
import pytesseract
import pyperclip
from PIL import Image, ImageGrab
import platform
import subprocess

# Tesseract path
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

monitor = None
running = True
DEBUG = True
Flag = 1

def check_setup():
    """Verify system setup."""
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    try:
        result = subprocess.run(["file", pytesseract.pytesseract.tesseract_cmd], 
                              capture_output=True, text=True, timeout=5)
        print(f"Tesseract: {result.stdout.strip()}")
    except:
        print("Tesseract: /usr/local/bin/tesseract")

def select_roi():
    """Select region with cv2.selectROI."""
    try:
        print("\nCapturing screen...")
        img_pil = ImageGrab.grab()
        img_array = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print("Draw rectangle around text. Press ENTER to confirm, ESC to cancel.")
        r = cv2.selectROI("Select region", img_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            return None
        
        return {"left": x, "top": y, "width": w, "height": h}
    except Exception as e:
        print(f"Error: {e}")
        return None

def grab_region(rect):
    """Grab region from screen."""
    try:
        bbox = (rect["left"], rect["top"], 
                rect["left"] + rect["width"], 
                rect["top"] + rect["height"])
        img = ImageGrab.grab(bbox=bbox)
        return np.array(img)
    except Exception as e:
        return None

def preprocess(img_array):
    """Convert image to grayscale and threshold."""
    try:
        if img_array is None or img_array.size == 0:
            return None
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, mode='RGB')
        else:
            pil_img = Image.fromarray(img_array)
        
        pil_gray = pil_img.convert('L')
        pil_bw = pil_gray.point(lambda x: 255 if x > 127 else 0, '1')
        return np.array(pil_bw)
    except Exception as e:
        return None

def normalize_text(t):
    """Clean up text."""
    try:
        t = t.replace('\x0c', ' ').strip()
        return ' '.join(t.split())
    except:
        return ""

def ocr_image(img):
    """Run OCR."""
    if img is None:
        return ""
    try:
        config = "--oem 3 --psm 6 -l eng"
        result = pytesseract.image_to_string(img, config=config, timeout=3)
        return result if result else ""
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract not found. Install: brew install tesseract")
        return ""
    except:
        return ""

def reselect():
    """Reselect ROI."""
    global monitor
    try:
        m = select_roi()
        if not m:
            if monitor:
                print("Keeping previous region.")
                return True
            return False
        monitor = m
        print(f"Selected: {monitor}")
        return True
    except:
        return bool(monitor)

def main():
    """Main OCR loop - runs until Ctrl+C"""
    global running, Flag
    
    print("\n=== Clipboard OCR ===\n")
    check_setup()
    
    if not reselect():
        print("Exiting.")
        return

    last_copied = ""
    frame_count = 0
    error_count = 0

    print("\nRunning OCR...")
    print("Press Ctrl+C to stop\n")

    try:
        while running:
            try:
                frame = grab_region(monitor)
                if frame is None or frame.size == 0:
                    error_count += 1
                    time.sleep(0.5)
                    continue
                
                proc = preprocess(frame)
                if proc is None:
                    error_count += 1
                    if error_count > 20:
                        print("Too many errors. Reselecting...")
                        reselect()
                        error_count = 0
                    time.sleep(0.5)
                    continue
                
                text = normalize_text(ocr_image(proc))
                error_count = 0
                frame_count += 1
                
                if DEBUG and frame_count % 20 == 0:
                    print(f"[{frame_count}] {'Text: ' + text if text else 'No text'}")
                
                if text and text != last_copied:
                    try:
                        pyperclip.copy(text)
                        last_copied = text
                        print(f"[COPIED] {text}")
                    except:
                        pass
                
                time.sleep(0.2)
                
            except Exception as e:
                error_count += 1
                if error_count > 20:
                    print("Too many errors. Stopping.")
                    break
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        print("Done.")

if __name__ == "__main__":
    main()