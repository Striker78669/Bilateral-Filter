import cv2
import numpy as np

def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))

def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

def entropy(image, x, y, diameter):
    histogram = np.zeros(256)
    
    for i in range(diameter):
        for j in range(diameter):
            n_x = x - (diameter // 2 - i)
            n_y = y - (diameter // 2 - j)
            if n_x >= len(image):
                n_x -= len(image)
            if n_y >= len(image[0]):
                n_y -= len(image[0])
            pixel_value = image[int(n_x)][int(n_y)]
            histogram[pixel_value] += 1
    
    entropy_val = 0
    total_pixels = diameter * diameter
    for count in histogram:
        if count > 0:
            probability = count / total_pixels
            entropy_val += -probability * np.log2(probability)
    
    return entropy_val

def bilateral_filter(image, diameter, sigma_i, sigma_s, sigma_e):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_pixel = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x = row - (diameter // 2 - k)
                    n_y = col - (diameter // 2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    ge = gaussian(entropy(image, row, col, diameter) - entropy(image, n_x, n_y, diameter), sigma_e)
                    wp = gi * gs * ge
                    filtered_pixel += image[int(n_x)][int(n_y)] * wp
                    wp_total += wp
            filtered_pixel = int(np.round(filtered_pixel / wp_total))
            new_image[row][col] = filtered_pixel

    return new_image

def on_mouse(event, x, y, flags, param):
    global image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        diameter = 100
        sigma_i = 20.0
        sigma_s = 20.0
        sigma_e = 20.0
        # Apply bilateral filter to the entire image
        filtered_image = bilateral_filter(image_gray, diameter, sigma_i, sigma_s, sigma_e)
        # Create a copy of the original image for display
        image_display = np.copy(image)
        # Calculate the coordinates of the corners of the rectangle
        x1 = max(0, x - diameter // 2)
        y1 = max(0, y - diameter // 2)
        x2 = min(image.shape[1] - 1, x + diameter // 2)
        y2 = min(image.shape[0] - 1, y + diameter // 2)
        # Replace the corresponding region in the display image with the filtered region
        image_display[y1:y2+1, x1:x2+1] = filtered_image[y1:y2+1, x1:x2+1]
        cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Save the result
        cv2.imwrite("result_image.png", image_display)

# Load the image
image = cv2.imread("img4.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_display = np.copy(image)

# Display the original image
cv2.imshow("Original Image", image_display)

# Set mouse callback function
cv2.setMouseCallback("Original Image", on_mouse)

# Wait for a key press and close all OpenCV windows when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
