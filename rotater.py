''' Rotating image based on the orientation of texts "OSD" '''
from pytesseract import Output
import pytesseract
import cv2
import os
import imutils
import matplotlib.pyplot as plt

# Setting Tesseract executable path
tesseract_path = r"C:\Users\intern-pradeep\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

class Rotater:
    def __init__(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image file not found.")
        
        self.image = cv2.imread(img_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.result = pytesseract.image_to_osd(self.gray, output_type=Output.DICT)
        print(self.result)
        print("Detected orientation: {}".format(self.result["orientation"]))
        print("Rotate by {} degrees to correct".format(self.result["rotate"]))

        # Rotate the image to correct the orientation
        self.rotated = imutils.rotate_bound(self.image, angle=self.result["rotate"])

        # Display the original and rotated images using Matplotlib
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(self.rotated, cv2.COLOR_BGR2RGB))
        plt.title("Rotated Image")
        plt.axis("off")

        plt.show()

        # Save the rotated image
        output_path = r'D:\Codes\CV\chola_working\output_images\Rotated.jpg'
        cv2.imwrite(output_path, self.rotated)
        print("Rotated image saved to:", output_path)

# File path of the image
filepath = r'D:\Codes\CV\chola_working\Input_images\IMG-20240201-WA0022.jpg'

# Create an instance of the Rotater class
rotater = Rotater(filepath)


'''for r,s,f in os.walk("/"):
    for i in f:
        if "tesseract" in i:
            print(os.path.join(r,i))

import pytesseract

# Setting Tesseract executable path
tesseract_path = r'/anaconda/envs/chumma/Library/bin/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path'''