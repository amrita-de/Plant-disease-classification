import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load the trained model
model = load_model("model/plant_disease_model.keras")

# Class labels
class_names = ['Apple_scab', 'Black_rot', 'Common_rust', 'Gray_leaf_spot', 'healthy']

def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class

    except Exception as e:
        return f"Error: {e}"

def predict_and_display_gui(img_path, image_label, prediction_label):
    predicted_class = predict_disease(img_path)
    prediction_label.config(text=f"Prediction: {predicted_class}")

    try:
        img = Image.open(img_path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", f"Error displaying image: {e}")

def upload_and_predict_gui():
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        predict_and_display_gui(file_path, image_label, prediction_label)

# Create the main window
root = tk.Tk()
root.title("Plant Disease Prediction")

# Image display label
image_label = tk.Label(root)
image_label.pack(pady=10)

# Prediction label
prediction_label = tk.Label(root, text="Prediction: ")
prediction_label.pack(pady=10)

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict_gui)
upload_button.pack(pady=10)

root.mainloop()