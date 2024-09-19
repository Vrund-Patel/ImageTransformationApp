import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import random

class ImageTransformationApp:
    def __init__(self):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        # Initialize filter library
        self.filters = {
            'Grayscale': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'Blur': self.custom_blur,
            'Edge Detection': self.custom_edge_detection,
            'Warm Tone': self.warm_tone,
            'Cool Tone': self.cool_tone,
            'Vibrant Colors': self.vibrant_colors,
            'Black and White': self.black_and_white,
            'Sepia': self.sepia,
            'Vintage': self.old_vintage,
            'Oil Painting': self.oil_painting,
            'Charcoal Sketch': self.charcoal_sketch,
            'HDR': self.hdr,
            'Glow Effect': self.glow_effect,
        }

        # Create a simple GUI
        self.root = tk.Tk()
        self.root.title("Image Transformation App")

        # Create labels for displaying original and transformed images
        self.original_label = tk.Label(self.root, text="Original Image")
        self.original_label.pack(side=tk.LEFT)
        self.transformed_label = tk.Label(self.root, text="Transformed Image")
        self.transformed_label.pack(side=tk.RIGHT)

        # Create dropdown menu for filters
        self.filter_var = tk.StringVar(self.root)
        self.filter_var.set(list(self.filters.keys())[0])  # Set default filter
        filter_menu = tk.OptionMenu(self.root, self.filter_var, *self.filters.keys())
        filter_menu.pack()

        # Create a scale for adjusting filter strength
        self.filter_strength_scale = tk.Scale(self.root, label="Filter Strength", from_=0, to=1, resolution=0.01,
                                              orient=tk.HORIZONTAL)
        self.filter_strength_scale.set(0.7)  # Set default strength
        self.filter_strength_scale.pack()

        # Create a variable to store the current frame
        self.current_frame = None

        # Start the main loop for updating the images
        self.update_images()

    def custom_blur(self, img):
        # Custom blur filter
        kernel = np.ones((15, 15), dtype=np.float32) / 225
        result = cv2.filter2D(img, -1, kernel)
        return result

    def custom_edge_detection(self, img):
        # Custom edge detection filter (Sobel operator)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        result = np.uint8(edge_mag)
        return result

    def capture_image(self):
        # Capture a single frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Couldn't capture frame.")
            return None
        return frame

    def apply_filter(self, image, filter_name):
        # Apply the selected filter to the image
        if filter_name in self.filters:
            filtered_image = self.filters[filter_name](image)

            # Check if the filtered image has only one channel (grayscale)
            if len(filtered_image.shape) == 2:
                # Convert the grayscale image to three channels
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

            return filtered_image
        else:
            print("Error: Filter not found.")
            return image

    def warm_tone(self, image):
        # Apply a warm tone to the image (golden tones)
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result[:, :, 0] = np.clip(result[:, :, 0] + 40, 0, 255)  # Increase blue channel
        result[:, :, 1] = np.clip(result[:, :, 1] + 20, 0, 255)  # Increase green channel
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def cool_tone(self, image):
        # Apply a cool tone to the image (bluish tones)
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result[:, :, 2] = np.clip(result[:, :, 2] + 40, 0, 255)  # Increase red channel
        result[:, :, 1] = np.clip(result[:, :, 1] + 20, 0, 255)  # Increase green channel
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def vibrant_colors(self, image):
        # Boost saturation for vibrant colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def black_and_white(self, image):
        # Convert image to black and white
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Apply a binary threshold to create a true black and white image
        _, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

        # Convert black and white image back to BGR (3 channels)
        img_bgr = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)

        return img_bgr

    def sepia(self, image):
        # Adjusted sepia tone to match color #704214
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = result.astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.393 + result[:, :, 1] * 0.769 + result[:, :, 2] * 0.189, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 0] * 0.349 + result[:, :, 1] * 0.686 + result[:, :, 2] * 0.168, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 0] * 0.272 + result[:, :, 1] * 0.534 + result[:, :, 2] * 0.131, 0, 255)

        # Adjust the intensity of the sepia tone
        intensity = 30
        slider_value = self.filter_strength_scale.get()
        max_intensity = 70  # Maximum intensity allowed
        intensity = min(intensity, max_intensity * (slider_value / 0.7))

        result = result + intensity

        # Adjust the balance between red, green, and blue channels to match color #704214
        result[:, :, 0] = 112
        result[:, :, 1] = 66
        result[:, :, 2] = 20

        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    def old_vintage(self, image):
        # Define vintage color levels
        VINTAGE_COLOR_LEVELS = {
            'r': [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 19, 19, 20, 21, 22, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 52, 54, 55, 57, 59, 60, 62, 65, 67, 69, 70, 72, 74, 77, 79, 81, 83, 86, 88, 90, 92, 94, 97, 99, 101, 103, 107, 109, 111, 112, 116, 118, 120, 124, 126, 127, 129, 133, 135, 136, 140, 142, 143, 145, 149, 150, 152, 155, 157, 159, 162, 163, 165, 167, 170, 171, 173, 176, 177, 178, 180, 183, 184, 185, 188, 189, 190, 192, 194, 195, 196, 198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 228, 229, 229, 230, 231, 232, 232, 233, 234, 234, 235, 236, 236, 237, 238, 238, 239, 239, 240, 241, 241, 242, 242, 243, 244, 244, 245, 245, 245, 246, 247, 247, 248, 248, 249, 249, 249, 250, 251, 251, 252, 252, 252, 253, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            'g' : [0, 0, 1, 2, 2, 3, 5, 5, 6, 7, 8, 8, 10, 11, 11, 12, 13, 15, 15, 16, 17, 18, 18, 19, 21, 22, 22, 23, 24, 26, 26, 27, 28, 29, 31, 31, 32, 33, 34, 35, 35, 37, 38, 39, 40, 41, 43, 44, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 113, 114, 115, 117, 118, 119, 120, 122, 123, 124, 126, 127, 128, 129, 131, 132, 133, 135, 136, 137, 138, 140, 141, 142, 144, 145, 146, 148, 149, 150, 151, 153, 154, 155, 157, 158, 159, 160, 162, 163, 164, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 186, 186, 187, 188, 189, 190, 192, 193, 194, 195, 195, 196, 197, 199, 200, 201, 202, 202, 203, 204, 205, 206, 207, 208, 208, 209, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 219, 219, 220, 221, 222, 223, 223, 224, 225, 226, 226, 227, 228, 228, 229, 230, 231, 232, 232, 232, 233, 234, 235, 235, 236, 236, 237, 238, 238, 239, 239, 240, 240, 241, 242, 242, 242, 243, 244, 245, 245, 246, 246, 247, 247, 248, 249, 249, 249, 250, 251, 251, 252, 252, 252, 253, 254, 255],
            'b' : [53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 61, 61, 61, 62, 62, 63, 63, 63, 64, 65, 65, 65, 66, 66, 67, 67, 67, 68, 69, 69, 69, 70, 70, 71, 71, 72, 73, 73, 73, 74, 74, 75, 75, 76, 77, 77, 78, 78, 79, 79, 80, 81, 81, 82, 82, 83, 83, 84, 85, 85, 86, 86, 87, 87, 88, 89, 89, 90, 90, 91, 91, 93, 93, 94, 94, 95, 95, 96, 97, 98, 98, 99, 99, 100, 101, 102, 102, 103, 104, 105, 105, 106, 106, 107, 108, 109, 109, 110, 111, 111, 112, 113, 114, 114, 115, 116, 117, 117, 118, 119, 119, 121, 121, 122, 122, 123, 124, 125, 126, 126, 127, 128, 129, 129, 130, 131, 132, 132, 133, 134, 134, 135, 136, 137, 137, 138, 139, 140, 140, 141, 142, 142, 143, 144, 145, 145, 146, 146, 148, 148, 149, 149, 150, 151, 152, 152, 153, 153, 154, 155, 156, 156, 157, 157, 158, 159, 160, 160, 161, 161, 162, 162, 163, 164, 164, 165, 165, 166, 166, 167, 168, 168, 169, 169, 170, 170, 171, 172, 172, 173, 173, 174, 174, 175, 176, 176, 177, 177, 177, 178, 178, 179, 180, 180, 181, 181, 181, 182, 182, 183, 184, 184, 184, 185, 185, 186, 186, 186, 187, 188, 188, 188, 189, 189, 189, 190, 190, 191, 191, 192, 192, 193, 193, 193, 194, 194, 194, 195, 196, 196, 196, 197, 197, 197, 198, 199]
        }

        def modify_all_pixels(im, pixel_callback):
            width, height = im.size
            pxls = im.load()
            for x in range(width):
                for y in range(height):
                    pxls[x, y] = pixel_callback(x, y, *pxls[x, y])

        def vintage_colors(im, color_map=VINTAGE_COLOR_LEVELS):
            r_map = color_map['r']
            g_map = color_map['g']
            b_map = color_map['b']

            def adjust_levels(x, y, r, g, b):
                return r_map[r], g_map[g], b_map[b]

            modify_all_pixels(im, adjust_levels)
            return im

        def add_noise(im, noise_level=20):
            def pixel_noise(x, y, r, g, b):
                noise = random.randint(0, noise_level) - noise_level / 2
                # Ensure that the resulting values are integers
                new_r = max(0, min(int(r + noise), 255))
                new_g = max(0, min(int(g + noise), 255))
                new_b = max(0, min(int(b + noise), 255))
                return new_r, new_g, new_b

            modify_all_pixels(im, pixel_noise)
            return im

        # Convert image to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply vintage colors
        vintage_image = vintage_colors(pil_image)

        # Add noise
        noisy_image = add_noise(vintage_image)

        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(noisy_image), cv2.COLOR_RGB2BGR)

        return result

    def oil_painting(self, image):
        # Apply oil painting filter with fine brush strokes
        return cv2.stylization(image, sigma_s=150, sigma_r=0.25)

    def charcoal_sketch(self, image):
        # Apply pencil sketch filter
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted_img = cv2.bitwise_not(img_gray)

        # Apply Gaussian blur to the inverted image
        blurred_img = cv2.GaussianBlur(inverted_img, (111, 111), 0)

        # Invert the blurred image
        inverted_blurred_img = cv2.bitwise_not(blurred_img)

        # Blend the inverted blurred image with the original color image
        pencil_sketch = cv2.divide(img_gray, inverted_blurred_img, scale=256.0)

        # Convert the pencil sketch to a BGR image
        sketch_bgr = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)

        return sketch_bgr

    def hdr(self, image):
        # Apply HDR (High Dynamic Range) filter
        return cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    def glow_effect(self, image):
        # Apply soft glow for a dreamy atmosphere
        blurred = cv2.GaussianBlur(image, (0, 0), 10)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def display_image(self, image, label):
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert image to PIL format
        image = Image.fromarray(image)
        # Convert PIL image to Tkinter PhotoImage
        photo = ImageTk.PhotoImage(image=image)
        # Update the label with the new image
        label.config(image=photo)
        label.image = photo

    def run(self):
        self.root.mainloop()

        # Release the webcam
        self.cap.release()

    def update_images(self):
        # Capture a frame
        self.current_frame = self.capture_image()

        # Update the original image label
        self.display_image(self.current_frame, self.original_label)

        # Apply and display the selected filter on the transformed image
        if self.current_frame is not None:
            selected_filter = self.filter_var.get()
            transformed_image = self.apply_filter(self.current_frame, selected_filter)

            # Resize transformed_image to match the dimensions of current_frame
            transformed_image = cv2.resize(transformed_image, (self.current_frame.shape[1], self.current_frame.shape[0]))

            # Ensure both images have the same data type
            self.current_frame = self.current_frame.astype(transformed_image.dtype)

            # Adjust filter strength based on the scale value
            filter_strength = self.filter_strength_scale.get()
            transformed_image = cv2.addWeighted(self.current_frame, 1 - filter_strength,
                                                transformed_image, filter_strength, 0)

            self.display_image(transformed_image, self.transformed_label)

        # Repeat the update after a delay (in milliseconds)
        self.root.after(10, self.update_images)

if __name__ == "__main__":
    app = ImageTransformationApp()
    app.run()
