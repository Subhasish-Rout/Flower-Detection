import os
import cv2
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data and image directories
data_dir = os.path.join(BASE_DIR, 'clean_data')
os.makedirs(data_dir, exist_ok=True)

# Check for 'Images' or 'images'
img_dir = os.path.join(BASE_DIR, 'Images')
if not os.path.exists(img_dir):
    img_dir = os.path.join(BASE_DIR, 'images')

if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Images directory not found at {img_dir}")

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def process_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return None
    file_path = os.path.join(img_dir, filename)
    img = cv2.imread(file_path)
    if img is None:
        return None
    # Resize and convert to grayscale
    img = cv2.resize(img, (100, 100))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = filename.split('_')[0].strip().lower()
    return img, label

all_files = os.listdir(img_dir)
print(f"Loading and processing {len(all_files)} images using multi-threading...")

images = []
labels = []

# High-speed parallel processing with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_file, all_files)
    for res in results:
        if res is not None:
            images.append(res[0])
            labels.append(res[1])

images = np.array(images, dtype=np.uint8)
labels = np.array(labels)

print(f"\nProcessing complete in record time!")
print(f"Total valid images loaded: {len(images)}")
print(f"Image array shape: {images.shape}")
print(f"Classes: {np.unique(labels)}")

# Fast binary save
images_path = os.path.join(data_dir, 'images.p')
labels_path = os.path.join(data_dir, 'labels.p')

with open(images_path, 'wb') as f:
    pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(labels_path, 'wb') as f:
    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Dataset successfully saved to '{data_dir}'!")
