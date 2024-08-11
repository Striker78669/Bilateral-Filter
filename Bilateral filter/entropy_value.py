import numpy as np
from skimage.measure import shannon_entropy
import skimage.io as io

def compute_entropy_from_image(img_path):
    img = io.imread(img_path)
    entropy = shannon_entropy(img)
    return entropy

# Example usage:
image_path = 'img2.bmp'
print(f"Image entropy: {compute_entropy_from_image(image_path):.4f}")
