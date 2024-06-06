import torch
from YoloClasses import Extractor
from ultralytics import YOLO


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging


# Set PaddleOCR logging to CRITICAL
logging.getLogger('ppocr').setLevel(logging.CRITICAL)
# document = torch.hub.load('yolov5', 'custom', path='models/document_best.pt', force_reload=True, source='local')
# document_model = Identifier(document)

v8_model_path = YOLO(r"models\detail_best.pt")
extracter = Extractor(v8_model_path)
