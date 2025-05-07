import cv2
import numpy as np
import joblib
from core.preprocess import load_image, denoise_image
from core.features import extract_features

# Load model
model = joblib.load("model.joblib")
print("✅ Model loaded from model.joblib")

# Load and process test image
img_path = "test_images/scanner_77.png"  # we have to give input for testing

img = load_image(img_path)
img_denoised = denoise_image(img)
features = extract_features(img_denoised).reshape(1, -1)

prediction = model.predict(features)[0]

# Map the predicted class to its label
label_map = {0: "Camera", 1: "Scanner"}

# Print the prediction
print(f"Predicted class: {label_map[prediction]}")

