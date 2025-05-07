

import cv2
import numpy as np
import pywt
from skimage.restoration import denoise_tv_chambolle

def load_image(path, size=(256, 256)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not load image: {path}")
    image = cv2.resize(image, size)
    return image

def denoise_image(image, wavelet='db4', level=3):
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    reconstructed = pywt.waverec2(coeffs, wavelet)
    tv_denoised = denoise_tv_chambolle(reconstructed, weight=0.1)
    return image - tv_denoised
