import cv2
import numpy

def gaussian(x, sigma):
    return (1.0 / (2 * numpy.pi * (sigma ** 2))) * numpy.exp(-(x ** 2) / (2 * (sigma ** 2)))

def distance(x1, y1, x2, y2):
    return numpy.sqrt(numpy.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

def entropy(image, x, y, diameter):
    histogram = numpy.zeros(256)
    
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
            entropy_val += -probability * numpy.log2(probability)
    
    return entropy_val

def bilateral_filter(image, diameter, sigma_i, sigma_s, sigma_e):
    new_image = numpy.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
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
                    filtered_image += image[int(n_x)][int(n_y)] * wp
                    wp_total += wp
            filtered_image = int(numpy.round(filtered_image / wp_total))
            new_image[row][col] = filtered_image

    return new_image

image = cv2.imread("img5 (2).jpg", 0)
filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
image_own = bilateral_filter(image, 7, 20.0, 20.0, 20.0)
cv2.imwrite("filtered_image_own.png", image_own)
