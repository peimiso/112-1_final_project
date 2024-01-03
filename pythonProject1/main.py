import cv2
import numpy as np

# Read images : src image will be cloned into dst
im = cv2.imread("li_3.png")
obj = cv2.imread("img_1.png")
obj_2 = cv2.imread("m.png")
obj=cv2.resize(obj,(int(im.shape[1]*0.13+3),int(im.shape[0]*0.55)), interpolation=cv2.INTER_AREA)
obj_2=cv2.resize(obj_2,(int(im.shape[1]*0.13+3),int(im.shape[0]*0.55)), interpolation=cv2.INTER_AREA)

# Create an all white mask
mask = 225* np.ones(obj.shape, obj.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (int(height / 2)+27,int( width / 2)+5)
print(center)

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
cv2.imshow("image", normal_clone)
cv2.waitKey(0)
mask_2 = 255* np.ones(obj_2.shape, obj_2.dtype)

normal_clone = cv2.seamlessClone(obj,normal_clone, mask_2, center, cv2.NORMAL_CLONE)
normal_clone = cv2.seamlessClone(obj_2,normal_clone, mask_2, center, cv2.NORMAL_CLONE)
# Write results
cv2.imshow("image", normal_clone)
cv2.waitKey(0)
cv2.imshow("image", mixed_clone)
cv2.waitKey(0)