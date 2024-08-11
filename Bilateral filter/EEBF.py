import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        self.selected_area = {'x': -1, 'y': -1, 'w': -1, 'h': -1}
        self.selection_in_progress = False
        self.input_image = cv2.imread(input_image_path)
        self.display_image = self.input_image.copy()

    def apply_bilateral_filter(self, sigma_color, sigma_space, sigma_entropy):
        x, y, w, h = self.selected_area['x'], self.selected_area['y'], self.selected_area['w'], self.selected_area['h']
        selected_region = self.input_image[y:y+h, x:x+w]

        # Calculate entropy of the selected region
        entropy = self.calculate_entropy(selected_region)

        # Apply bilateral filter with entropy as an additional factor
        filtered_area = cv2.bilateralFilter(selected_region, 9, sigma_color, sigma_space)
        self.input_image[y:y+h, x:x+w] = filtered_area
        return self.input_image, selected_region

    def calculate_entropy(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist = hist.ravel() / hist.sum()
        non_zero_bins = hist[hist != 0]
        entropy = -np.sum(non_zero_bins * np.log2(non_zero_bins))
        return entropy

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
                    filtered_image, selected_region = self.apply_bilateral_filter(sigma_color=75, sigma_space=75, sigma_entropy=0.1)
                    temp_image = filtered_image.copy()
                    cv2.rectangle(temp_image, (self.selected_area['x'], self.selected_area['y']),
                                  (self.selected_area['x'] + self.selected_area['w'], self.selected_area['y'] + self.selected_area['h']),
                                  (0, 255, 0), 2)
                    cv2.imwrite('Filtered_Image.jpg', temp_image)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = ImageProcessor('img3 (1).jpg')
    processor.run()
