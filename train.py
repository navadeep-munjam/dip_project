import os
import numpy as np
from core.preprocess import load_image, denoise_image
from core.features import extract_features
from core.model import train_model, save_model
from tqdm import tqdm

def load_dataset(folder, label):
    X, y = [], []
    for f in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        path = os.path.join(folder, f)
        try:
            img = load_image(path)
            noise = denoise_image(img)
            features = extract_features(noise)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error with {f}: {e}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X1, y1 = load_dataset("images/scanner", 1)
    X2, y2 = load_dataset("images/camera", 0)

    X = np.vstack([X1, X2])

    y = np.concatenate([y1, y2])

    model = train_model(X, y)
    save_model(model)
    print("Model trained and saved.")