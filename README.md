# LUNG-X-RAY-PNEUMONIA-DETECTION-USING-COMPUTER-VISION-DEEP-LEARNING-
Pneumonia is a serious infection requiring fast, accurate diagnosis. This project uses a fine-tuned MobileNetV2 model to classify chest X-ray images as Normal or Pneumonia. A Tkinter GUI lets users upload images and get real-time predictions with confidence scores, offering a quick and user-friendly diagnostic tool.
This project implements an automated pneumonia detection system using deep learning and computer vision. A pre-trained MobileNetV2 model is fine-tuned to classify chest X-ray images as Normal or Pneumonia.
A simple Tkinter GUI is also included, allowing users to upload an X-ray image and receive real-time predictions with confidence scores.

⭐ Features

✓ MobileNetV2 transfer learning

✓ Class imbalance handled using class weights

✓ Fine-tuning for better performance

✓ Evaluation using accuracy, precision, recall & F1-score

✓ Tkinter GUI for real-time prediction

✓ Saves trained model (.h5 format)

✓ Training history plots (accuracy & loss)

📁 Project Structure
├── train_pneumonia_model.py      # Training & evaluation code
├── gui_app.py                    # Tkinter GUI for prediction
├── pneumonia_model_fast.h5       # Saved trained model
├── chest_xray/                   # Dataset (train/val/test folders)
└── README.md

📥 Dataset

Use the publicly available Chest X-ray Pneumonia dataset (Kaggle):
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia



🔧 Installation
1. Install Required Libraries
pip install tensorflow numpy matplotlib opencv-python pillow scikit-learn


Tkinter comes pre-installed with Python.

🚀 How to Run the Project
1. Train the Model
python train_pneumonia_model.py

2. Start the GUI
python gui_app.py
