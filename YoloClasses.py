'''This code self.captures aadhar card from live video and bound it with a ounding box and cuts around the box '''



import cv2
import torch
import numpy as np
import time
import easyocr
from ultralytics import YOLO


class Identifier():
    def __init__(self,model_path):
        # initializing webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.last_bbox = None
        self.stable_duration = 1.0  # Minimum duration (in seconds) for the bounding box to remain stable
        self.start_stable_time = None
        while True:
            self.ret, self.img = self.cap.read()

            if self.ret:

                # Run document model inference
                self.document_model_result = model_path(self.img)

                # Extract bounding box coordinates from YOLO result
                self.bboxes = self.document_model_result.xyxy[0].cpu().numpy()
                
                if len(self.bboxes) > 0:
                    # Take the first bounding box detected by YOLO
                    self.bbox = self.bboxes[0]
                    
                    # Check if the bounding box is stable
                    if self.last_bbox is not None and np.allclose(self.bbox, self.last_bbox, atol=10):
                        # If the bounding box is stable for self.stable_duration seconds, self.capture the image
                        if self.start_stable_time is None:
                            self.start_stable_time = time.time()
                        elif time.time() - self.start_stable_time >= self.stable_duration:
                            # Crop the region from the original image
                            self.x_min, self.y_min, self.x_max, self.y_max = self.bbox[:4]
                            self.cropped_img = self.img[int(self.y_min):int(self.y_max), int(self.x_min):int(self.x_max)]

                            # Display the cropped image
                            cv2.imshow('Cropped Image', self.cropped_img)

                            # Save the cropped image
                            cv2.imwrite('cropped_image.jpg', self.cropped_img)

                            # Reset stability check variables
                            self.start_stable_time = None
                            self.last_bbox = None
                    else:
                        self.last_bbox = self.bbox
                        self.start_stable_time = None

                # Render the results
                self.img_rendered = self.document_model_result.render()
                self.img_rendered = self.img_rendered[0]

                # Display the frame with results
                cv2.imshow('Webcam', self.img_rendered)

                # Check for key press
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break

                # Release resources
        self.cap.release()
        cv2.destroyAllWindows()





class Extractor():
    def __init__(self,V8_model):
        self.reader = easyocr.Reader(['en'])

        # Load image
        self.img_path = r'D:\Codes\CV\cropped_image.jpg'
        self.img = cv2.imread(self.img_path)

        # Model
        self.model = V8_model

        # Object classes
        self.classNames = [
            "aadhaar number", "address", "dob", "emblem", "father", "gender", "goi symbol",
            "issue date", "logo", "name", "photo", "uiai icon", "uiai symbol", "vid", "yob"
        ]

        # Classes for which OCR will be performed
        self.ocr_classes = ["aadhaar number", "dob", "name", "gender", "photo"]

        # Perform object detection
        self.results = self.model(self.img, stream=True)
        # Keep track of the classes for which OCR has been performed
        self.ocr_performed = set()

        # Iterate over the results
        for self.r in self.results:
            self.boxes = self.r.boxes
            for self.box in self.boxes:
                # Bounding box coordinates
                self.x1, self.y1, self.x2, self.y2 = map(int, self.box.xyxy[0])

                # Class name
                self.cls = int(self.box.cls[0])
                self.class_name = self.classNames[self.cls]

                # Perform OCR only for specified classes and if not already performed
                if self.class_name in self.ocr_classes and self.class_name not in self.ocr_performed:
                    # Add the class to the set of classes for which OCR has been performed
                    self.ocr_performed.add(self.class_name)

                    # Crop the region of interest (ROI) for OCR
                    self.roi = self.img[self.y1:self.y2, self.x1:self.x2]

                    # Resize the ROI for better OCR accuracy
                    self.roi_resized = cv2.resize(self.roi, (0, 0), fx=2, fy=2)  # Example resizing by a factor of 2

                    # Perform OCR using EasyOCR
                    self.ocr_results = self.reader.readtext(self.roi_resized)

                    # Extract text from OCR results
                    self.ocr_text = ' '.join([self.result[1] for self.result in self.ocr_results])

                    # Print OCR result
                    print(f"OCR Text for {self.class_name}: {self.ocr_text.strip()}")

                    # Save photo if detected
                    if self.class_name == "photo":
                        # Define the file path to save the cropped photo
                        self.photo_path = 'cropped_1.jpg'

                        # Save the cropped photo
                        cv2.imwrite(self.photo_path, self.roi)

                        # Display the cropped photo
                        self.cropped_img = cv2.imread(self.photo_path)
                        cv2.imshow('Cropped Photo', self.cropped_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        # Display the original image without bounding boxes
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


