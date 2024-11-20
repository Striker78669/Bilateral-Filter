import cv2
import numpy as np

def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))

def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            entropy_term = 0  # Initialize entropy term

            for k in range(diameter):
                for l in range(diameter):
                    n_x = row - (diameter / 2 - k)
                    n_y = col - (diameter / 2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])

                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image += image[int(n_x)][int(n_y)] * wp
                    wp_total += wp

                    # Calculate entropy term
                    entropy_term += wp * np.log2(wp + 1e-10)  # Add a small value to avoid log(0)

            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(np.round(filtered_image + entropy_term))  # Add entropy term

    return new_image

image = cv2.imread("img2.bmp", 0)
filtered_image_own = bilateral_filter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_own_with_entropy.png", filtered_image_own)
