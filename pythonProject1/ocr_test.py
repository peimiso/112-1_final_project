import pytesseract
from PIL import Image
import cv2
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = cv2.imread("li_2.png",cv2.IMREAD_COLOR)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img=cv2.threshold(img,110,255,cv2.THRESH_BINARY_INV)[1]
img=cv2.medianBlur(img,3)
kernel=np.ones((2,2),np.uint8)
img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow("img",img)
cv2.waitKey(0)
print(pytesseract.image_to_string(img, lang="eng",config="r`--oem 3 --psm 6"))