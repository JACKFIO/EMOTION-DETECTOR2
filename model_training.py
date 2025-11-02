"""
Face Emotion Detection Model Training Script
This script trains a CNN model to recognize 7 emotions from facial images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# ==================== CONFIGURATION ====================
# IMPORTANT: Update this path to where your FER2013 dataset is located
DATASET_PATH = 'fer2013'  # Change this if your dataset is elsewhere

# Image parameters
IMG_SIZE = 48  # FER2013 images are 48x48 pixels
BATCH_SIZE = 64
EPOCHS = 30  # Start with 30 epochs (increase if needed)

# Emotion labels (7 classes)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("="*60)
print("🎭 FACE EMOTION DETECTION - MODEL TRAINING")
print("="*60)
print(f"\n📁 Dataset path: {DATASET_PATH}")
print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE} pixels")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🔄 Training epochs: {EPOCHS}")
print(f"😊 Emotions to detect: {', '.join(EMOTIONS)}")
print("="*60 + "\n")

# ==================== DATA PREPARATION ====================
print("📂 Step 1: Loading and preparing data...")

# Create data generators for training and validation
# Data augmentation helps the model learn better by creating variations of images
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1
    rotation_range=10,            # Randomly rotate images
    width_shift_range=0.1,       # Randomly shift images horizontally
    height_shift_range=0.1,      # Randomly shift images vertically
    horizontal_flip=True,        # Randomly flip images
    zoom_range=0.1,              # Randomly zoom images
    validation_split=0.2         # Use 20% of training data for validation
)

# For test data, we only rescale (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
print("📥 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

# Load validation data (from training set)
print("📥 Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Load test data
print("📥 Loading test data...")
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

print(f"\n✅ Data loaded successfully!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Classes detected: {train_generator.class_indices}\n")

# ==================== MODEL ARCHITECTURE ====================
print("🏗️  Step 2: Building the CNN model...")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Output Layer (7 emotions)
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ==================== CALLBACKS ====================
# Early stopping: stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# ==================== TRAINING ====================
print("\n🎯 Step 3: Training the model...")
print("⏳ This may take 30-60 minutes depending on your computer...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✅ Training completed!")

# ==================== EVALUATION ====================
print("\n📊 Step 4: Evaluating the model...")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🎯 Test Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

# ==================== SAVE MODEL ====================
print("\n💾 Step 5: Saving the model...")

model.save('face_emotionModel.h5')
print("✅ Model saved as 'face_emotionModel.h5'")

# ==================== PLOT TRAINING HISTORY ====================
print("\n📈 Step 6: Generating training plots...")

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training plots saved as 'training_history.png'")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📦 Output files created:")
print(f"   ✓ face_emotionModel.h5 - Trained model")
print(f"   ✓ training_history.png - Training graphs")
print(f"\n🎯 Final Results:")
print(f"   ✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   ✓ Total Epochs Trained: {len(history.history['accuracy'])}")
print("\n✨ Your model is ready to use in the Flask app!")
print("="*60 + "\n")