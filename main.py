import torch
from YoloClasses import Extractor,Identifier
from ultralytics import YOLO



document = torch.hub.load('yolov5', 'custom', path='models/document_best.pt', force_reload=True, source='local')
document_model = Identifier(document)

v8_model_path = YOLO(r"D:\Codes\CV\models\detail_best.pt")
extracter = Extractor(v8_model_path)
