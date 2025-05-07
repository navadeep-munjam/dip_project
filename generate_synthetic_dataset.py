import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import random

def add_text(image, text="Sample Text", position=(20, 20), font_size=20):
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=0)
    return np.array(pil_img)

def apply_artifacts(image, blur=True, rotation=True, noise_level=10):
    if rotation:
        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return np.clip(image, 0, 255).astype(np.uint8)

def generate_scanner_image(shape=(256, 256), save_path=None):
    image = np.ones(shape, dtype=np.uint8) * 255
    for i in range(5):
        text = f"Document Line {random.randint(1000,9999)}"
        image = add_text(image, text=text, position=(10, 20 + i * 40), font_size=18)

    # Add scan artifacts
    image = apply_artifacts(image, blur=True, rotation=True, noise_level=5)

    if save_path:
        cv2.imwrite(save_path, image)

def generate_camera_image(shape=(256, 256), save_path=None):
    image = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * random.randint(200, 255)
    for i in range(3):
        text = f"Photo Note {random.randint(100,999)}"
        image = add_text(image, text=text, position=(15, 30 + i * 60), font_size=22)

    image = apply_artifacts(image, blur=True, rotation=True, noise_level=10)

    if save_path:
        cv2.imwrite(save_path, image)

def create_dataset(n_per_class=1000, output_dir="images", start_index=100):
    os.makedirs(f"{output_dir}/scanner", exist_ok=True)
    os.makedirs(f"{output_dir}/camera", exist_ok=True)

    print(f"Generating {n_per_class} scanner images...")
    for i in range(start_index, start_index + n_per_class):
        generate_scanner_image(save_path=f"{output_dir}/scanner/scanner_{i}.png")

    print(f"Generating {n_per_class} camera images...")
    for i in range(start_index, start_index + n_per_class):
        generate_camera_image(save_path=f"{output_dir}/camera/camera_{i}.png")

    print("✅ Advanced synthetic dataset generated successfully.")

if __name__ == "__main__":
    create_dataset(n_per_class=10000)
