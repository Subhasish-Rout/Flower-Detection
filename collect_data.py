import os
import cv2
import time
import numpy as np

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'Images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# 1. Select Flower Category
print("========================================")
print("     FLOWER DATA COLLECTION SYSTEM      ")
print("========================================")
flower_name = input("Enter flower class (daisy/dandelion/rose/sunflower/tulip) [rose]: ").strip().lower()
if not flower_name:
    flower_name = "rose"

# 2. Camera Setup (Auto-detects Webcam or prompts for Phone IP Webcam)
def get_camera_stream():
    # Try local webcam first
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            print("Detected active local webcam (Camera Index 0).")
            return test_cap

    test_cap.release()
    print("\nNo local webcam detected.")
    ip_input = input("Enter phone IP or Stream URL [default: http://192.168.5.143:8080/video]: ").strip()
    if not ip_input:
        ip_input = "http://192.168.5.143:8080/video"
    elif not ip_input.startswith("http"):
        ip_input = f"http://{ip_input}:8080/video"

    print(f"Connecting to: {ip_input} ...")
    cap = cv2.VideoCapture(ip_input)
    if not cap.isOpened():
        # Fallback to /shot.jpg if /video fails
        fallback_url = ip_input.replace("/video", "/shot.jpg")
        print(f"Retrying with snapshot URL: {fallback_url} ...")
        cap = cv2.VideoCapture(fallback_url)
    return cap

cap = get_camera_stream()
if not cap.isOpened():
    print("Error: Could not open any camera stream. Please check camera connection/URL.")
    exit(1)

# Find starting image index to avoid overwriting existing images
existing = [f for f in os.listdir(IMAGE_DIR) if f.startswith(flower_name + "_")]
start_idx = len(existing) + 1

target_count = 100
captured = []
auto_capture = False
capture_interval = 2  # Capture every 2 frames in auto mode for super fast collection
frame_count = 0
fps_timer = time.time()
fps = 0

print(f"\nReady! Collecting {target_count} images for class '{flower_name}'")
print("Controls:")
print("  - Press 'c' or SPACEBAR : Capture 1 frame")
print("  - Press 'a'             : Toggle TURBO AUTO-CAPTURE (collects 100 images in seconds)")
print("  - Press 'q' or ESC      : Stop & save immediately\n")

cv2.namedWindow("Flower Data Collection", cv2.WINDOW_NORMAL)

while len(captured) < target_count:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera feed lost or unavailable.")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # Central flower targeting reticle
    box_size = min(h, w) // 2
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Calculate real-time FPS
    if frame_count % 15 == 0:
        now = time.time()
        fps = 15.0 / max(now - fps_timer, 0.001)
        fps_timer = now

    display = frame.copy()
    reticle_color = (0, 0, 255) if auto_capture else (0, 255, 0)
    cv2.rectangle(display, (x1, y1), (x2, y2), reticle_color, 2)

    # UI Information Header
    status = "AUTO-CAPTURING..." if auto_capture else "MANUAL (Press 'c' or 'a')"
    cv2.putText(display, f"Class: {flower_name.upper()} | Progress: {len(captured)}/{target_count}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(display, f"Mode: {status} | FPS: {fps:.1f}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, reticle_color, 2)

    # Visual progress bar at bottom
    bar_width = int((len(captured) / target_count) * (w - 40))
    cv2.rectangle(display, (20, h - 30), (w - 20, h - 15), (50, 50, 50), -1)
    cv2.rectangle(display, (20, h - 30), (20 + bar_width, h - 15), (0, 255, 0), -1)

    cv2.imshow("Flower Data Collection", display)
    key = cv2.waitKey(1) & 0xFF

    should_capture = False
    if key in (ord('c'), ord(' ')):
        should_capture = True
    elif key == ord('a'):
        auto_capture = not auto_capture
    elif key in (27, ord('q')):
        print("Collection stopped early by user.")
        break

    if auto_capture and (frame_count % capture_interval == 0):
        should_capture = True

    if should_capture:
        roi = frame[y1:y2, x1:x2].copy()
        captured.append(roi)

cv2.destroyAllWindows()
cap.release()

# Fast bulk save to disk
if captured:
    print(f"\nSaving {len(captured)} images to '{IMAGE_DIR}'...")
    for i, img in enumerate(captured):
        out_name = f"{flower_name}_ ({start_idx + i}).jpg"
        cv2.imwrite(os.path.join(IMAGE_DIR, out_name), img)
    print(f"Success! {len(captured)} images saved in '{IMAGE_DIR}'.")
else:
    print("No images were captured.")
