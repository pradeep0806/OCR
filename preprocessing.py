''' basic preprocessing steps on images like thresholding, resizing and grayscaling'''
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

filepath = r"D:\Codes\CV\chola_working\output_images\Rotated.jpg"



class preprocess:
    def __init__(self,image:str):
        self.img = cv2.imread(image)
        self.img = cv2.resize(self.img,(1280,1024))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        #plt.imshow(img,cmap='gray')
        #plt.show()
        #ret, thresh1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        self.thresh2 = cv2.adaptiveThreshold(self.img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)

        plt.imshow(self.thresh2,cmap='gray')
        plt.show()
        cv2.imwrite('D:\Codes\CV\chola_working\output_images\prep.jpg',self.thresh2)




image = preprocess(filepath)

#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.imshow(th3,cmap='gray')
#plt.show()

