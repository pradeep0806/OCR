import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import logging
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI

app = FastAPI()

# Disable logging for PaddleOCR
logging.getLogger('ppocr').setLevel(logging.CRITICAL)

# Alternatively, set the environment variable to control verbosity
os.environ["PPOCR_LOG_LEVEL"] = "ERROR"

def initialize_paddleocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

def load_image(img_path):
    return cv2.imread(img_path)

def get_class_names():
    return [
        "aadhaar number", "address", "dob", "emblem", "father", "gender", "goi symbol",
        "issue date", "logo", "name", "photo", "uiai icon", "uiai symbol", "vid", "yob"
    ]

def perform_object_detection(model, img):
    return model(img, stream=True)

def crop_and_resize(img, x1, y1, x2, y2, resize_factor=2):
    roi = img[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (0, 0), fx=resize_factor, fy=resize_factor)
    return roi_resized

def extract_text_from_ocr(ocr_results):
    if ocr_results and len(ocr_results) > 0:
        return ' '.join([result[1][0] for result in ocr_results[0]])
    return ""

def save_and_display_image(roi, filename='cropped_1.jpg'):
    cv2.imwrite(filename, roi)
    cropped_img = cv2.imread(filename)
    cv2.imshow('Cropped Photo', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(V8_model, img_path):
    reader = initialize_paddleocr()
    img = load_image(img_path)
    class_names = get_class_names()
    ocr_classes = {"aadhaar number", "dob", "name", "gender", }
    ocr_performed = set()

    results = perform_object_detection(V8_model, img)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = class_names[cls]

            if class_name in ocr_classes and class_name not in ocr_performed:
                ocr_performed.add(class_name)
                roi_resized = crop_and_resize(img, x1, y1, x2, y2)

                # Display the ROI before OCR


                ocr_results = reader.ocr(roi_resized, cls=True)
                ocr_text = extract_text_from_ocr(ocr_results)

                print(f"OCR Text for {class_name}: {ocr_text.strip()}")

    
# Example usage (assuming you have a V8 model instance):
model = YOLO(r'models/detail_best.pt')

@app.post("/")
async def end(items:str):
    
    main(model, r'Input_images/IMG-20240201-WA0022.jpg')
    
