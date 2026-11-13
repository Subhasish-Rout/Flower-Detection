import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Setup paths relative to script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "clean_data", "label_classes.p")

# 1. Load Model & Class Labels
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'!\n"
        "Please run 'train.py' first to train and generate 'final_model.h5'."
    )

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Compile / warmup graph for blazing fast inference
_warmup = np.zeros((1, 100, 100, 1), dtype=np.float32)
_ = model(_warmup, training=False)

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "rb") as f:
        labels = pickle.load(f)
else:
    labels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

print(f"Loaded Classes ({len(labels)}): {labels}")

# 2. Camera Setup (Auto-detects Webcam or prompts for Phone IP Webcam)
def get_camera_stream():
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            print("Using local webcam (Camera Index 0).")
            return test_cap

    test_cap.release()
    print("\nNo local webcam detected.")
    ip_input = input("Enter phone IP or Stream URL [default: http://192.168.5.143:8080/video]: ").strip()
    if not ip_input:
        ip_input = "http://192.168.5.143:8080/video"
    elif not ip_input.startswith("http"):
        ip_input = f"http://{ip_input}:8080/video"

    print(f"Connecting to video stream: {ip_input} ...")
    cap = cv2.VideoCapture(ip_input)
    if not cap.isOpened():
        fallback = ip_input.replace("/video", "/shot.jpg")
        print(f"Retrying with snapshot URL: {fallback} ...")
        cap = cv2.VideoCapture(fallback)
    return cap

cap = get_camera_stream()
if not cap.isOpened():
    print("Error: Could not connect to camera.")
    exit(1)

# Preprocessing: 100x100 Grayscale normalized to [0, 1]
def preprocess_roi(roi):
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (100, 100))
    roi = roi.astype(np.float32) * (1.0 / 255.0)
    return roi.reshape(1, 100, 100, 1)

print("\nStarting live flower detection (Turbo Mode)...")
print("Instructions:")
print("  - Position the flower inside the center box.")
print("  - Press 'q' or ESC to exit.\n")

cv2.namedWindow("Flower Detection", cv2.WINDOW_NORMAL)

frame_count = 0
fps_timer = time.time()
fps = 0
last_class = "Scanning..."
last_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera stream ended or lost.")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # Target region in center of frame
    box_size = min(h, w) // 2
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # High-speed inference: run model call directly (5x faster than model.predict)
    # Perform prediction every 2nd frame to keep video rendering ultra-fluid
    if frame_count % 2 == 0:
        flower_roi = frame[y1:y2, x1:x2]
        processed = preprocess_roi(flower_roi)
        probs = model(processed, training=False).numpy()[0]
        pred_idx = int(np.argmax(probs))
        last_conf = float(probs[pred_idx]) * 100.0
        last_class = str(labels[pred_idx]).upper()

    # Calculate real-time FPS
    if frame_count % 15 == 0:
        now = time.time()
        fps = 15.0 / max(now - fps_timer, 0.001)
        fps_timer = now

    # UI Rendering
    color = (0, 255, 0) if last_conf >= 60 else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Class Badge above box
    badge_text = f"{last_class} ({last_conf:.1f}%)"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, badge_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Header HUD
    cv2.putText(frame, f"Flower Detection | FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Flower Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
cap.release()
print("Live detection ended.")
