import cv2
from skimage.metrics import structural_similarity as ssim

# Load the images
image1 = cv2.imread('img.bmp')
image2 = cv2.imread('filtered_image_OpenCV.png')

# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate SSIM
ssim_index, _ = ssim(gray_image1, gray_image2, full=True)

# Print the SSIM index
print(f'SSIM Index: {ssim_index}')
