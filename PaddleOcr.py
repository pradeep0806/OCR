''' a general ocr for image and can even detect tamil and english text, gives output as image with texts on side and bounding boxes'''
from paddleocr import PaddleOCR,draw_ocr

from datetime import datetime
start_time = datetime.now()

ocr_tamil = PaddleOCR(lang='ta')

# Initialize PaddleOCR for English language
ocr_english = PaddleOCR(lang='en')

img_path = 'D:\Codes\CV\chola_working\output_images\prep.jpg'



# Perform OCR for English text
result_english = ocr_english.ocr(img_path, cls=False)

print("\nEnglish Text:")
for idx in range(len(result_english)):
    res = result_english[idx]
    for line in res:
        text = line[1][0]  # Get the detected text
        print(text)  # Print the detected text to the console
# draw result
from PIL import Image
result = result_english[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='D:\Codes\CV\simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save(r'result.jpg')
endtime = datetime.now()

print("time taken : ",endtime-start_time)

'''import os, sys

print(sys.executable)
print(os.getenv('CONDA_PREFIX'))'''