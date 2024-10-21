import cv2
import numpy as np
from matplotlib import pyplot as plt

kernel = np.ones((3, 3), np.uint8) 

class helper:
    def __init__(self):
        return

    def dilate(self, img, iters=1):
        return cv2.dilate(img, kernel, iterations=iters)

    def erode(self, img, iters=1):
        return cv2.erode(img, kernel, iterations=iters)

    def houghLines(self, img):
        img = cv2.GaussianBlur(img,(5,5),0)
        ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        edges = cv2.Canny(gray, 120, 240)
        # Detect points that form a line
        lines = cv2.HoughLinesP(gray, 3, np.pi/720, 1000, minLineLength=200, maxLineGap=40)
        return lines
    
    def satAdjHSV(self, imgBGR, sAdj):
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imgHSV.astype('int'))
        s = s*sAdj
        s = np.clip(s,0,255)
        return cv2.merge([h,s,v]).astype('uint8')
    
    def fieldMask(self, imgHSV):
        (h, s, v) = cv2.split(imgHSV)       
        ret, mask0 = cv2.threshold(h,40,255,cv2.THRESH_BINARY)
        ret, mask1 = cv2.threshold(h,70,255,cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask0, mask0, mask=mask1)
        mask = self.erode(mask, 14)
        mask = self.dilate(mask, 50)
        return mask