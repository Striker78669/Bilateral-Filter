import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        self.selected_area = {'x': -1, 'y': -1, 'w': -1, 'h': -1}
        self.selection_in_progress = False
        self.input_image = cv2.imread(input_image_path)
        self.display_image = self.input_image.copy()

    def apply_bilateral_filter(self, sigma_color=75, sigma_space=75, min_kernel_size=5, max_kernel_size=15):
        x, y, w, h = self.selected_area['x'], self.selected_area['y'], self.selected_area['w'], self.selected_area['h']
        selected_region = self.input_image[y:y+h, x:x+w]

        # Calculate entropy of the selected region
        entropy = self.calculate_entropy(selected_region)

        # Normalize entropy to a range of [0, 1]
        max_entropy = np.log2(256)  # Maximum possible entropy for an 8-bit image
        normalized_entropy = entropy / max_entropy

        # Calculate adaptive kernel size based on entropy
        adaptive_kernel_size = int(min_kernel_size + normalized_entropy * (max_kernel_size - min_kernel_size))

        # Ensure kernel size is odd and within a valid range
        if adaptive_kernel_size % 2 == 0:
            adaptive_kernel_size += 1

        # Apply bilateral filter with adaptive kernel size
        filtered_area = cv2.bilateralFilter(selected_region, adaptive_kernel_size, sigma_color, sigma_space)
        
        self.input_image[y:y+h, x:x+w] = filtered_area
        return self.input_image, selected_region

    def calculate_entropy(self, image):
        # Calculate entropy for the given region
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist = hist.ravel() / hist.sum()
        non_zero_bins = hist[hist != 0]
        entropy = -np.sum(non_zero_bins * np.log2(non_zero_bins))
        return entropy

    def apply_spatially_adaptive_filter(self, block_size=32, min_kernel_size=5, max_kernel_size=15, sigma_color=75, sigma_space=75):
        height, width = self.input_image.shape[:2]

        # Loop over the image in blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = self.input_image[y:y+block_size, x:x+block_size]

                # Calculate entropy for the block
                entropy = self.calculate_entropy(block)

                # Normalize entropy and calculate adaptive kernel size
                max_entropy = np.log2(256)
                normalized_entropy = entropy / max_entropy
                adaptive_kernel_size = int(min_kernel_size + normalized_entropy * (max_kernel_size - min_kernel_size))

                if adaptive_kernel_size % 2 == 0:
                    adaptive_kernel_size += 1

                # Apply bilateral filter on the block
                filtered_block = cv2.bilateralFilter(block, adaptive_kernel_size, sigma_color, sigma_space)
                self.input_image[y:y+block_size, x:x+block_size] = filtered_block

        return self.input_image

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_area['x'], self.selected_area['y'] = x, y
            self.selection_in_progress = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_area['w'], self.selected_area['h'] = x - self.selected_area['x'], y - self.selected_area['y']
            self.selection_in_progress = False

    def run(self):
        cv2.namedWindow('Input Image')
        cv2.setMouseCallback('Input Image', self.mouse_callback)

        while True:
            cv2.imshow('Input Image', self.display_image)

            if self.selection_in_progress:
                temp_image = self.display_image.copy()
                cv2.rectangle(temp_image, (self.selected_area['x'], self.selected_area['y']),
                              (self.selected_area['x'] + self.selected_area['w'], self.selected_area['y'] + self.selected_area['h']),
                              (0, 255, 0), 2)
                cv2.imshow('Input Image', temp_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                if self.selected_area['w'] > 0 and self.selected_area['h'] > 0:
                    # Apply adaptive bilateral filter to selected area
                    filtered_image, selected_region = self.apply_bilateral_filter()
                    temp_image = filtered_image.copy()
                    cv2.rectangle(temp_image, (self.selected_area['x'], self.selected_area['y']),
                                  (self.selected_area['x'] + self.selected_area['w'], self.selected_area['y'] + self.selected_area['h']),
                                  (0, 255, 0), 2)
                    cv2.imwrite('Filtered_Image.jpg', temp_image)
            elif key == ord('a'):
                # Apply spatially adaptive filtering to the entire image
                filtered_image = self.apply_spatially_adaptive_filter()
                cv2.imshow('Filtered Image', filtered_image)
                cv2.imwrite('Spatially_Adaptive_Filtered_Image.jpg', filtered_image)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = ImageProcessor('img3 (1).jpg')
    processor.run()