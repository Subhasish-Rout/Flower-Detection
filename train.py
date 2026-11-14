import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'clean_data')
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.h5')
CLASSES_PATH = os.path.join(DATA_DIR, 'label_classes.p')

print("=== Loading Flower Dataset ===")
images_file = os.path.join(DATA_DIR, 'images.p')
labels_file = os.path.join(DATA_DIR, 'labels.p')

if not os.path.exists(images_file) or not os.path.exists(labels_file):
    raise FileNotFoundError(f"Missing dataset in {DATA_DIR}. Run consolidated.py first!")

with open(images_file, 'rb') as f:
    images = pickle.load(f)

with open(labels_file, 'rb') as f:
    labels = pickle.load(f)

print(f"Loaded {len(images)} images.")
print(f"Original image shape: {images.shape}")

# Preprocess images: normalize to [0, 1] and add channel dimension (100, 100, 1)
images = images.astype('float32') / 255.0
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

# Encode string labels to integers
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
class_names = list(encoder.classes_)
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Save class names mapping for deploy.py
with open(CLASSES_PATH, 'wb') as f:
    pickle.dump(class_names, f)
print(f"Saved class labels to: {CLASSES_PATH}")

# Split dataset into train and validation sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# Build CNN Architecture
model = models.Sequential([
    layers.Input(shape=(100, 100, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

print("\n=== Training Model ===")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")

# Ensure final model is saved
model.save(MODEL_PATH)
print(f"Successfully saved final model to: {MODEL_PATH}")

