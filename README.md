# 🌸 Real-Time Flower Detection & Classification System

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

An end-to-end, real-time flower detection and classification application built with **OpenCV** and **Convolutional Neural Networks (CNN)** using **TensorFlow/Keras**. 

The system supports custom data collection, high-speed multi-threaded preprocessing, automated model training, and low-latency live video classification via local webcam or smartphone IP camera streams.

---

## 🎯 Supported Flower Classes

The default model recognizes 5 distinct flower species:
* 🌼 **Daisy**
* 🌾 **Dandelion**
* 🌹 **Rose**
* 🌻 **Sunflower**
* 🌷 **Tulip**

---

## ✨ Key Features

- **⚡ Blazing Fast Live Inference**: Optimized direct TensorFlow execution achieving **~18 ms/frame** (~5x faster than standard `.predict()`).
- **📷 Flexible Camera Source**: Auto-detects local USB webcams and provides seamless fallback to smartphone IP webcam streams (e.g., via IP Webcam Android app).
- **🚀 Multi-Threaded Dataset Preprocessing**: `ThreadPoolExecutor` parallelizes image ingestion, processing over 4,000 images in under 10 seconds.
- **📸 Turbo Data Collection**: Includes manual (`Space`/`c`) and burst auto-capture (`a`) modes with on-screen reticles and live progress meters.
- **🧠 Custom CNN Architecture**: Multi-stage feature extraction with Batch Normalization, Max Pooling, Dropout regularization, and Softmax classification.

---

## 📁 Project Structure

```plaintext
Flower Detection/
│
├── Images/                     # Raw image dataset categorized by prefix (e.g., rose_ (1).jpg)
├── clean_data/                 # Preprocessed and serialized binary datasets
│   ├── images.p                # Pickled normalized image array
│   ├── labels.p                # Pickled class labels array
│   └── label_classes.p         # Label encoder class names
│
├── collect_data.py             # Interactive camera tool to capture new flower samples
├── consolidated.py             # Multi-threaded image loader, resizer, and preprocessor
├── train.py                    # CNN model training script with EarlyStopping & ModelCheckpoint
├── deploy.py                   # Real-time live camera classification HUD
├── final_model.h5              # Trained Keras CNN weights
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/flower-detection.git
cd flower-detection
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: (Optional) Collect New Flower Data
Capture your own flower images using your webcam or phone camera:
```bash
python collect_data.py
```
* Enter flower class name (e.g., `rose`, `daisy`).
* **Controls**:
  * Press `c` or `SPACE`: Capture 1 frame.
  * Press `a`: Toggle **TURBO AUTO-CAPTURE** (captures 100 images in seconds).
  * Press `q` or `ESC`: Stop and save.

---

### Step 2: Consolidate & Preprocess Dataset
Preprocess the images (resizes to $100 \times 100$, converts to grayscale, packages labels):
```bash
python consolidated.py
```
* Uses multi-threading to process the entire dataset in ~8 seconds.
* Outputs saved to `clean_data/images.p` and `clean_data/labels.p`.

---

### Step 3: Train the Convolutional Neural Network
Train the CNN on the preprocessed dataset:
```bash
python train.py
```
* Automatically splits dataset (80% train / 20% validation).
* Employs `EarlyStopping` and saves the best model to `final_model.h5`.

---

### Step 4: Real-Time Live Detection
Launch live video detection:
```bash
python deploy.py
```
* **Auto-detects** your local webcam. If no USB camera is detected, simply enter your phone's IP Webcam stream URL (e.g. `http://192.168.1.X:8080/video`).
* Aim the flower at the center targeting box.
* The HUD displays:
  * Predicted flower name
  * Confidence score percentage
  * Real-time FPS counter
* Press `q` or `ESC` to quit.

---

## 📊 Model Architecture

```
Layer (type)                     Output Shape          Param #
=================================================================
conv2d (Conv2D)                  (None, 100, 100, 32)  320
batch_normalization              (None, 100, 100, 32)  128
max_pooling2d                    (None, 50, 50, 32)    0
_________________________________________________________________
conv2d_1 (Conv2D)                (None, 50, 50, 64)    18,496
batch_normalization_1            (None, 50, 50, 64)    256
max_pooling2d_1                  (None, 25, 25, 64)    0
_________________________________________________________________
conv2d_2 (Conv2D)                (None, 25, 25, 128)   73,856
batch_normalization_2            (None, 25, 25, 128)   512
max_pooling2d_2                  (None, 12, 12, 128)   0
_________________________________________________________________
flatten                          (None, 18432)         0
dense                            (None, 128)           2,359,424
dropout                          (None, 128)           0
dense_1 (Dense - Softmax)        (None, 5)             645
=================================================================
Total params: 2,453,637 (9.36 MB)
```

---

## 📜 License

This project is licensed under the MIT License - feel free to use and adapt it for your own computer vision projects!

