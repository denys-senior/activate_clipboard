#!/usr/bin/env python3
"""
Clipboard OCR for macOS Intel
Captures selected screen region and performs OCR
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageEnhance
import pytesseract

# Environment setup for Intel Mac compatibility
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Global to store selected region
selected_region = None


def test_imagegrab():
    """Test PIL ImageGrab functionality"""
    print("[TEST] Testing PIL ImageGrab...")
    try:
        img = ImageGrab.grab()
        print(f"[TEST] ImageGrab OK: {img.size}")
        return True
    except Exception as e:
        print(f"[TEST] ImageGrab failed: {e}")
        return False


def test_cv2():
    """Test OpenCV functionality"""
    print("[TEST] Testing OpenCV...")
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("[TEST] OpenCV OK")
        return True
    except Exception as e:
        print(f"[TEST] OpenCV failed: {e}")
        return False


def select_region_interactive():
    """
    Let user select screen region with cv2.selectROI
    Falls back to manual coordinate input if selectROI fails
    """
    print("[SELECT] Capturing screenshot for region selection...")
    try:
        screenshot = ImageGrab.grab()
        img_array = np.array(screenshot)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        print("[SELECT] Opening region selection window...")
        print("[SELECT] Drag to select region, press SPACE to confirm, ESC to cancel")

        try:
            roi = cv2.selectROI("Select Region", img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            if roi[2] == 0 or roi[3] == 0:  # width or height is 0
                print("[SELECT] Invalid selection (empty region)")
                return get_manual_coordinates()

            region = {
                'left': int(roi[0]),
                'top': int(roi[1]),
                'width': int(roi[2]),
                'height': int(roi[3])
            }
            print(f"[SELECT] Region selected: {region}")
            return region
        except Exception as e:
            print(f"[SELECT] selectROI failed: {e}")
            cv2.destroyAllWindows()
            return get_manual_coordinates()
    except Exception as e:
        print(f"[SELECT] Screenshot capture failed: {e}")
        return get_manual_coordinates()


def get_manual_coordinates():
    """Get region coordinates manually from user"""
    print("[SELECT] Falling back to manual coordinate input...")
    try:
        left = int(input("Enter left coordinate: "))
        top = int(input("Enter top coordinate: "))
        width = int(input("Enter width: "))
        height = int(input("Enter height: "))

        region = {
            'left': left,
            'top': top,
            'width': width,
            'height': height
        }
        print(f"[SELECT] Manual region: {region}")
        return region
    except Exception as e:
        print(f"[SELECT] Manual input failed: {e}")
        # Default region
        return {'left': 100, 'top': 100, 'width': 800, 'height': 600}


def preprocess(pil_img):
    """
    Preprocess image for OCR using adaptive thresholding
    Improved version that preserves text contrast
    """
    # Convert to grayscale
    gray_img = pil_img.convert('L')
    gray_array = np.array(gray_img)

    # Step 1: Enhance contrast to make text stand out
    # Apply histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_array)

    # Step 2: Denoise
    denoised = cv2.medianBlur(enhanced, 3)

    # Step 3: Adaptive thresholding instead of fixed threshold
    # This adjusts threshold locally, better for varying lighting conditions
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size (must be odd)
        2    # Constant subtracted from mean
    )

    # Step 4: Optional: Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    print(f"[PREPROCESS] Input shape: {gray_array.shape}, Output shape: {morph.shape}")
    print(f"[PREPROCESS] Pixel range: min={morph.min()}, max={morph.max()}, mean={morph.mean():.1f}")

    return morph


def capture_region(region):
    """Capture the specified screen region"""
    try:
        box = (
            region['left'],
            region['top'],
            region['left'] + region['width'],
            region['top'] + region['height']
        )
        captured = ImageGrab.grab(bbox=box)
        return captured
    except Exception as e:
        print(f"[CAPTURE] Failed to capture region: {e}")
        return None


def ocr_image(img_array, use_gpu=False):
    """
    Perform OCR on image using Tesseract
    Improved configuration for better text detection
    """
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(img_array, np.ndarray):
            pil_img = Image.fromarray(img_array)
        else:
            pil_img = img_array

        # Tesseract configuration
        # --oem 1: Use LSTM engine (better for modern text)
        # --psm 6: Assume single uniform block of text
        config = '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:\'"'

        text = pytesseract.image_to_string(pil_img, config=config)

        if text.strip():
            print(f"[OCR] Detected text: {text[:100]}")
        else:
            print("[OCR] No text detected (empty result)")

        return text.strip()
    except Exception as e:
        print(f"[OCR] Error during OCR: {e}")
        return ""


def show_debug_image(region_img, preprocessed_img):
    """
    Display original and preprocessed images side-by-side for debugging
    """
    try:
        # Convert to display format
        if isinstance(region_img, Image.Image):
            orig_array = np.array(region_img)
            if len(orig_array.shape) == 3 and orig_array.shape[2] == 3:
                orig_bgr = cv2.cvtColor(orig_array, cv2.COLOR_RGB2BGR)
            else:
                orig_bgr = orig_array
        else:
            orig_bgr = region_img

        # Resize to fit side-by-side
        height = 400
        aspect_orig = orig_bgr.shape[1] / orig_bgr.shape[0]
        width_orig = int(height * aspect_orig)
        orig_resized = cv2.resize(orig_bgr, (width_orig, height))

        # Resize preprocessed image
        preproc_resized = cv2.resize(preprocessed_img, (width_orig, height))
        preproc_color = cv2.cvtColor(preproc_resized, cv2.COLOR_GRAY2BGR)

        # Combine side-by-side
        combined = np.hstack([orig_resized, preproc_color])

        # Add labels
        cv2.putText(combined, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Preprocessed", (width_orig + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Debug: Original vs Preprocessed", combined)
        print("[DEBUG] Showing debug image (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[DEBUG] Failed to show debug image: {e}")


def main():
    """Main OCR loop"""
    print("=" * 60)
    print("macOS Intel Clipboard OCR")
    print("=" * 60)

    # Run tests
    if not test_imagegrab():
        print("[ERROR] ImageGrab test failed")
        sys.exit(1)

    if not test_cv2():
        print("[ERROR] OpenCV test failed")
        sys.exit(1)

    # Select region
    print("\n[MAIN] Starting region selection...")
    region = select_region_interactive()

    print("\n[MAIN] Starting OCR loop...")
    print("[MAIN] Press Ctrl+C to stop")

    first_iteration = True
    consecutive_empty = 0
    max_empty_frames = 5  # Show debug after 5 empty frames

    try:
        while True:
            # Capture region
            region_img = capture_region(region)
            if region_img is None:
                print("[MAIN] Failed to capture region, retrying...")
                time.sleep(1)
                continue

            # Preprocess
            preprocessed = preprocess(region_img)

            # Perform OCR
            text = ocr_image(preprocessed)

            # Track empty detections
            if not text:
                consecutive_empty += 1

                # Show debug image after several empty frames to help diagnose
                if consecutive_empty == max_empty_frames and first_iteration:
                    print(f"\n[DEBUG] Showing preprocessing output (frame #{consecutive_empty})...")
                    show_debug_image(region_img, preprocessed)
                    first_iteration = False
            else:
                consecutive_empty = 0
                first_iteration = False

            # Print timestamp
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Frame processed - Text: {'(empty)' if not text else 'FOUND'}")

            # Small delay to avoid CPU spinning
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[MAIN] Stopping OCR...")
        cv2.destroyAllWindows()
        print("[MAIN] Done")


if __name__ == '__main__':
    main()