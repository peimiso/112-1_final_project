import cv2
import numpy as np
from imutils import contours

image = cv2.imread('li_4.jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
thresh = cv2.adaptiveThreshold(V, 255,
cv2.ADAPTIVE_THRESH_MEAN_C,
cv2.THRESH_BINARY_INV, 17, 3)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#thresh= cv2.erode(thresh, kernel)
#thresh= cv2.dilate(thresh, kernel)
thresh = cv2.medianBlur(thresh,3)

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="left-to-right")
ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area < 800 and area > 200:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - thresh[y:y+h, x:x+w]
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        cv2.imshow('ROI_{}.png', ROI)

        ROI_number += 1

cv2.imshow('mask', mask)
cv2.imshow('thresh', thresh)
cv2.waitKey()