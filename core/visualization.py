
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_image, denoise_image
from features import extract_features
import pywt
from skimage.restoration import denoise_tv_chambolle
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Added missing import


def create_visualization_folder():
    """Create folder for saving visualizations if it doesn't exist"""
    if not os.path.exists('visualized_images'):
        os.makedirs('visualized_images')
        print("Created 'visualized_images' folder")
    return 'visualized_images'

def process_image_folder(input_folder):
    """Process all images in the input folder"""
    output_folder = create_visualization_folder()
    
    # Get all image files from input folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in {input_folder} with supported formats {valid_extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        try:
            img_path = os.path.join(input_folder, img_file)
            print(f"\nProcessing: {img_file}")
            
            # Generate visualization
            fig = visualize_pipeline(img_path)
            
            # Save visualization
            output_path = os.path.join(output_folder, f"vis_{os.path.splitext(img_file)[0]}.png")
            fig.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

def visualize_pipeline(image_path):
    """Create visualization for a single image"""
    # Load and process image
    original = load_image(image_path)
    noise_residual = denoise_image(original)
    features = extract_features(noise_residual)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Image Analysis: {os.path.basename(image_path)}", fontsize=14)
    
    # Original Image
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # Noise Residual
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
    im2 = ax2.imshow(noise_residual, cmap='gray')
    ax2.set_title('Noise Residual')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Features Plot
    # Features Plot
    ax3 = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    bars = ax3.bar(range(len(features)), features)
    ax3.set_title('Extracted Features')
    ax3.set_xticks(range(len(features)))
    
    feature_names = [
        'Row Corr Stats', 'Col Corr Stats', 'Freq Domain Features',
        'Row/Col Ratio', 'Freq Mean', 'Freq Var',
        'Freq Mean (Left)', 'Freq Mean (Top)'
    ]
    
    # Ensure label count matches feature count
    if len(features) != len(feature_names):
        feature_names = [f'Feature {i+1}' for i in range(len(features))]
    
    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add correlation pattern mini-plots
    M, N = noise_residual.shape
    row_ref = np.mean(noise_residual, axis=0)
    col_ref = np.mean(noise_residual, axis=1)
    row_corr = np.array([corr(noise_residual[i, :], row_ref) for i in range(M)])
    col_corr = np.array([corr(noise_residual[:, j], col_ref) for j in range(N)])
    
    # Insert correlation plots
    ax4 = plt.subplot2grid((3, 4), (0, 3))
    ax4.plot(row_corr, 'b-', alpha=0.7)
    ax4.set_title('Row Correlations')
    ax4.set_ylim(-1, 1)
    
    ax5 = plt.subplot2grid((3, 4), (1, 3))
    ax5.plot(col_corr, 'r-', alpha=0.7)
    ax5.set_title('Column Correlations')
    ax5.set_ylim(-1, 1)
    
    plt.tight_layout()
    return fig

def corr(x, y):
    """Helper function to calculate correlation"""
    x, y = x - np.mean(x), y - np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <input_folder>")
        print("Example: python visualize.py ./images")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)
    
    process_image_folder(input_folder)
    print("\nProcessing complete!")