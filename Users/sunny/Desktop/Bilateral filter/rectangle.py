import cv2
import numpy as np

# Global variables to store the coordinates of the selected area
selected_area = {'x': -1, 'y': -1, 'w': -1, 'h': -1}
selection_in_progress = False

def apply_bilateral_filter(selected_area, sigma_color, sigma_space, input_image):
    # Extract the selected area
    x, y, w, h = selected_area['x'], selected_area['y'], selected_area['w'], selected_area['h']
    selected_region = input_image[y:y+h, x:x+w]

    # Apply bilateral filter to the selected area
    filtered_area = cv2.bilateralFilter(selected_region, 9, sigma_color, sigma_space)

    # Replace the selected area in the input image with the filtered area
    filtered_image = input_image.copy()
    filtered_image[y:y+h, x:x+w] = filtered_area

    return filtered_image, selected_region, filtered_area

def mouse_callback(event, x, y, flags, param):
    global selected_area, selection_in_progress

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_area['x'], selected_area['y'] = x, y
        selection_in_progress = True

    elif event == cv2.EVENT_LBUTTONUP:
        selected_area['w'], selected_area['h'] = x - selected_area['x'], y - selected_area['y']
        selection_in_progress = False

def main():
    # Read the input image
    input_image = cv2.imread('img5.jpg')

    # Create a copy of the input image for display purposes
    display_image = input_image.copy()

    # Create a window and set the mouse callback function
    cv2.namedWindow('Input Image')
    cv2.setMouseCallback('Input Image', mouse_callback)

    while True:
        # Display the input image
        cv2.imshow('Input Image', display_image)

        # Draw rectangle for the selected area if selection is in progress
        if selection_in_progress:
            temp_image = display_image.copy()
            cv2.rectangle(temp_image, (selected_area['x'], selected_area['y']),
                          (selected_area['x'] + selected_area['w'], selected_area['y'] + selected_area['h']),
                          (0, 255, 0), 2)
            cv2.imshow('Input Image', temp_image)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('f'):  # Press 'f' to apply filter
            if selected_area['w'] > 0 and selected_area['h'] > 0:
                filtered_image, selected_part, filtered_part = apply_bilateral_filter(selected_area, sigma_color=75, sigma_space=75, input_image=input_image)
                cv2.imwrite('Filtered Image.jpg', filtered_image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
