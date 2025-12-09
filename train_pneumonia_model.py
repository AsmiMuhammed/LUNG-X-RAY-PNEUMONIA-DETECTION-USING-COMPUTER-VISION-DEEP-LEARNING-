# train_pneumonia_model_fast.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# --- Hide TensorFlow logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "chest_xray"
IMG_SIZE = (128, 128)       # smaller image → faster training
BATCH_SIZE = 32             # larger batch → fewer steps
EPOCHS = 5                  # fewer epochs
FINE_TUNE_EPOCHS = 3
SEED = 42

# -------------------------------
# Data Preparation
# -------------------------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# -------------------------------
# Compute Class Weights (optional but helps balance)
# -------------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

# -------------------------------
# Model (Simplified MobileNetV2)
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Train (Base Phase)
# -------------------------------
print("\n🚀 Starting Quick Training...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# Fine-tuning (Unfreeze last 10 layers)
# -------------------------------
print("\n🔧 Quick Fine-Tuning...")
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_gen,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# Evaluation
# -------------------------------
y_true = test_gen.classes
preds = model.predict(test_gen)
y_pred = (preds > 0.5).astype("int32").ravel()

print("\n✅ Test Accuracy:", round(np.mean(y_pred == y_true) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# -------------------------------
# Save and Verify
# -------------------------------
MODEL_NAME = "pneumonia_model_fast.h5"
try:
    model.save(MODEL_NAME)
    print(f"\n✅ Model saved as '{MODEL_NAME}'")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")

if os.path.exists(MODEL_NAME):
    print("📦 Saved model found successfully!")
else:
    print("⚠️ Model not found — check your path.")

# -------------------------------
# Plot Accuracy & Loss
# -------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.show()
