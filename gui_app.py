import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import os

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "pneumonia_model_fast.h5"
IMG_SIZE = (128, 128)

# Try loading the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("Pneumonia Detection ") 
root.geometry("550x650")
root.configure(bg="#e9f3f8")

tk.Label(root, text="Lung X-Ray Pneumonia Detection",
         font=("Helvetica", 16, "bold"), bg="#e9f3f8", fg="#003366").pack(pady=15)

img_label = tk.Label(root, bg="#e9f3f8")
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e9f3f8")
result_label.pack(pady=20)

# -------------------------------
# Functions
# -------------------------------
def browse_image():
    """Open a file dialog to select an X-ray image"""
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if path:
        img = Image.open(path).resize((320, 320))
        tk_img = ImageTk.PhotoImage(img)
        img_label.configure(image=tk_img)
        img_label.image = tk_img
        img_label.path = path
        result_label.config(text="")

def predict_image():
    """Run pneumonia prediction"""
    if model is None:
        messagebox.showerror("Model Error", "Model not loaded. Please check model path.")
        return
    if not hasattr(img_label, "path"):
        messagebox.showwarning("Input Error", "Please upload an X-ray image first.")
        return

    try:
        path = img_label.path
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img / 255.0, axis=0)

        pred = model.predict(img)[0][0]
        label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confidence = pred if pred > 0.5 else 1 - pred
        conf_percent = confidence * 100

        color = "red" if label == "PNEUMONIA" else "green"
        result_label.config(
            text=f"Prediction: {label}\nConfidence: {conf_percent:.2f}%",
            fg=color
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error: {e}")

# -------------------------------
# Buttons
# -------------------------------
tk.Button(root, text="Browse X-Ray", command=browse_image,
          font=("Arial", 12), bg="#007acc", fg="white", width=15).pack(pady=5)
tk.Button(root, text="Predict", command=predict_image,
          font=("Arial", 12), bg="#28a745", fg="white", width=15).pack(pady=10)

tk.Label(root, text="Developed using Computer Vision & Deep Learning",
         font=("Arial", 10, "italic"), bg="#e9f3f8", fg="#555").pack(side="bottom", pady=10)

root.mainloop()

