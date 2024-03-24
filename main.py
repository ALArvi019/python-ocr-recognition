import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from io import BytesIO
import pytesseract
import time

# IP camera configuration
url = "http://192.168.0.4/mjpeg/snap.cgi?chn=0"
username = 'admin'
password = '123456'

# Coordinates to crop the temperature image
x_coords_to_crop_temp = [990, 1043]
y_coords_to_crop_temp = [570, 606]

# Coordinates to crop the actual temperature image
x_coords_to_crop_actual_temp = [982, 993]
y_coords_to_crop_actual_temp = [600, 609]

rotation_angle = 353

def get_image():
    # Make HTTP request with basic authentication credentials
    response = requests.get(url, auth=HTTPDigestAuth(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        # Read the image from the response content
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    else:
        print("Error getting image from IP camera.")
        return None

def crop_image(img, x_coords, y_coords):
    cropped_img = img[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
    return cropped_img

def rotate_image(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    return rotated_img

def count_white_pixels(img):
    white_pixels = cv2.countNonZero(cv2.inRange(img, (200, 200, 200), (255, 255, 255)))
    return white_pixels

# Loop until the number of white pixels is greater than 15
while True:
    # Get the image
    img = get_image()

    if img is not None:
        # Crop the actual part of the image
        cropped_img_actual_temp = crop_image(img, x_coords_to_crop_actual_temp, y_coords_to_crop_actual_temp)

        # Count the white pixels
        white_pixels = count_white_pixels(cropped_img_actual_temp)

        if white_pixels > 15:
            break
        else:
            print("Waiting for one second...")
            time.sleep(1)

print("Number of white pixels is greater than 15. Proceeding with the code...")

# Crop the image
cropped_img = crop_image(img, x_coords_to_crop_temp, y_coords_to_crop_temp)

# Rotate the image
rotated_img = rotate_image(cropped_img, rotation_angle)

# Convert the image to grayscale
gray_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
_, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Add a 10px border to the image
threshold_img = cv2.copyMakeBorder(threshold_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Apply OCR
text = pytesseract.image_to_string(threshold_img, config='--psm 6  -c tessedit_char_whitelist=0123456789$')
print(text)

# Show the image
cv2.imshow("Image", threshold_img)
cv2.waitKey(0)

# save original image
cv2.imwrite('original_image.jpg', img)

# save temperature image
cv2.imwrite('temperature_image.jpg', threshold_img)

# save actual temperature image
cv2.imwrite('actual_temperature_image.jpg', cropped_img_actual_temp)
