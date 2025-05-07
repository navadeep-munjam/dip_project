import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft2
from skimage.feature import graycomatrix, graycoprops

def corr(x, y):
    x, y = x - np.mean(x), y - np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

def extract_features(noise):
    M, N = noise.shape
    row_ref = np.mean(noise, axis=0)
    col_ref = np.mean(noise, axis=1)
    row_corr = [corr(noise[i, :], row_ref) for i in range(M)]
    col_corr = [corr(noise[:, j], col_ref) for j in range(N)]

    def stats(x): 
        return [np.mean(x), np.median(x), np.max(x), np.min(x), np.var(x), kurtosis(x), skew(x)]

    features = stats(row_corr) + stats(col_corr)
    features.append(np.mean(row_corr) / (np.mean(col_corr) + 1e-8))

    # Frequency domain features
    freq_mag = np.abs(fft2(noise))
    freq_features = [
        np.mean(freq_mag),
        np.var(freq_mag),
        np.mean(freq_mag[:, :N//2]),
        np.mean(freq_mag[:M//2, :])
    ]
    
    # GLCM features
    quantized = (noise * 255).astype(np.uint8)
    glcm = graycomatrix(quantized, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    glcm_features = [contrast, homogeneity, energy, correlation]

    return np.array(features + freq_features + glcm_features)

# from scipy.stats import kurtosis, skew
# from scipy.fft import fft2
# import numpy as np

# def corr(x, y):
#     x, y = x - np.mean(x), y - np.mean(y)
#     return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

# def extract_features(noise):
#     M, N = noise.shape
#     row_ref = np.mean(noise, axis=0)
#     col_ref = np.mean(noise, axis=1)
#     row_corr = [corr(noise[i, :], row_ref) for i in range(M)]
#     col_corr = [corr(noise[:, j], col_ref) for j in range(N)]

#     def stats(x): 
#         return [np.mean(x), np.median(x), np.max(x), np.min(x), np.var(x), kurtosis(x), skew(x)]

#     features = stats(row_corr) + stats(col_corr)
#     features.append(np.mean(row_corr) / (np.mean(col_corr) + 1e-8))
#     # Compute the magnitude of the 2D Fourier Transform of the noise
#     freq_mag = np.abs(fft2(noise))
#     freq_features = [
#         np.mean(freq_mag),
#         np.var(freq_mag),
#         np.mean(freq_mag[:, :N//2]),
#         np.mean(freq_mag[:M//2, :])
#     ]
#     return np.array(features + freq_features)
