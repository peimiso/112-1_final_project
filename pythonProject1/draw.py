# import the necessary packages
import cv2
import imutils
import argparse
import pytesseract
import collections, numpy
import glob
import random

from PIL import Image, ImageDraw
# load the tic-tac-toe image and convert it to grayscale
ap = argparse.ArgumentParser()
pic_array=(glob.glob("C:/Users\Lenovo\PycharmProjects\pythonProject1/items\*"))
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
img_temp=image.copy()
img_temp_2=image.copy()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(args["image"])

V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
thresh = cv2.adaptiveThreshold(V, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY_INV, 17, 4)
thresh = cv2.medianBlur(thresh,5)
#cv2.imshow("GRAY", thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
output2 = cv2.erode(thresh, kernel)     # 先侵蝕，將白色小圓點移除
#output2 = cv2.dilate(output2, kernel)    # 再膨脹，白色小點消失
output2 = cv2.medianBlur(output2,3)
cv2.imshow("resut_K.png", output2)
cv2.waitKey()
# find all contours on the tic-tac-toe board
cnts = cv2.findContours(output2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
seg=[]
seg_reg=[]


height=image.shape[0]
weight=image.shape[1]

# loop over the contours
for (i, c) in enumerate(cnts): # compute the area of the contour along with the bounding box # to compute the aspect ratio
 area = (cv2.contourArea(c) )
 if (area == 0):
  continue
 (x, y, w, h) = cv2.boundingRect(c)
 hull = cv2.convexHull(c)
 hullArea = cv2.contourArea(hull)
 solidity = area / float(hullArea)
# initialize the character text
 if  (y<height*0.5 and h+y>height*0.5) and w<weight*0.2 and h<height*0.9 and  w*h>=0.015*height*weight and h>height*0.3 and h+y<height*0.95 and area/(h*w)>0.2:
  #cv2.drawContours(image, [c], -1, (255,0,0), -1)
  cv2.rectangle(img_temp, (x, y), (x + w, y + h), (255, 0, 0), 1)
  #print("(Contour #{}) -- solidity={:.2f}".format(i + 1, solidity))
  #print([c])
  seg.append((cv2.boundingRect(c),[c]))
  #cv2.imshow("a",image[y:y+h,x:x+w])
 # cv2.waitKey(0)
 # plate+=pytesseract.image_to_string(image[y:y+h,x:x+w], lang='eng', config='--psm 10')
 # show the output image
#(seg, seg_reg) = sort_contours(seg)
#print((seg))
#con='--oem 1 --psm 10 '
seg=sorted(seg, key = lambda x : x[0][0])
#for i in seg:
# word=output2[i[0][1]-5:i[0][1]+i[0][3]+5,i[0][0]-5:i[0][0]+i[0][2]+5]
 #cv2.imshow("Output2",word)
 #cv2.waitKey(0)
#print(plate)
#image.getcolors(image.size[0]*image.size[1])
#cv2.drawContours(image,seg[5][1], -1, (255,255,255), -1)

cv2.imshow("the char that be detected", img_temp)
cv2.waitKey()
if len(seg)==0:
 cv2.imwrite(args["image"],image)

if len(seg) !=1 or len(seg) !=0  :
 print("input indexs of the char to replace")
 #index = input()
 index="0 2 3"
 index=index.split(" ")
 print(int(index[1]))
 for i in index:
  i=int(i)
  char_color=image[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]].copy()
  cv2.drawContours(image,seg[i][1], -1, (255, 255, 255), -1)
  mask=image[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]].copy()
  cv2.drawContours(image,seg[i][1], -1, (255, 255,255), 2)
  #挖與前面的空格複製
  if i!=0 and seg[i][0][0]-seg[i-1][0][0]-seg[i-1][0][2]>=1:
   mid=seg[i][0][0]-seg[i-1][0][0]-seg[i-1][0][2]
   sticker=img_temp_2[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]-3:seg[i][0][0]-1]
   sticker=cv2.resize(sticker,(seg[i][0][2],seg[i][0][3]),interpolation=cv2.INTER_CUBIC)

   #sticker=cv2.bitwise_and(sticker,image[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]])

   #img_temp_2.paste(sticker, (seg[i][0][0],seg[i][0][1]))
   img_temp_2[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]] = sticker
   #img_temp_2[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]] = cv2.inpaint(img_temp_2[seg[i][0][1]:seg[i][0][1]+seg[i][0][3],seg[i][0][0]:seg[i][0][0]+seg[i][0][2]],)
   kernel = numpy.ones((4, 4), numpy.float32) / 16

   img_temp_2[seg[i][0][1]-1:seg[i][0][1]+1 + seg[i][0][3], seg[i][0][0]-1:seg[i][0][0] + seg[i][0][2]+1]= cv2.filter2D(img_temp_2[seg[i][0][1]-1:seg[i][0][1]+1 + seg[i][0][3], seg[i][0][0]-1:seg[i][0][0] + seg[i][0][2]+1], -1, kernel)
   cv2.imshow("or",img_temp_2)
   cv2.waitKey(0)
   #buliding mask to get char color

   hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
   lo = numpy.array([0, 0, 0])
   hi = numpy.array([254, 254, 254])
   mask_2 = cv2.inRange(hsv, lo, hi)
   mask[mask_2 > 0] = ( 0, 0, 0)#黑底白字
   mask=cv2.bitwise_not(mask)#白底黑字
   #cv2.imshow("or", mask)
   #cv2.waitKey(0)
   char_color=cv2.bitwise_or(char_color,mask) #色底色字->白底色字
   #cv2.imshow("or", char_color)
   #cv2.waitKey(0)
   #把白底扣掉找色字的平均
   char_color = char_color.reshape(-1, 3)
   unique, counts = numpy.unique(char_color, axis=0, return_counts=True)
   #

   print(unique[numpy.argmax(counts)])
   if unique[numpy.argmax(counts)][0]==255 and unique[numpy.argmax(counts)][1]==255 and unique[numpy.argmax(counts)][2]==255  :
     #print(unique[numpy.argmax(counts)])
     counts[numpy.argmax(counts)]=0
     unique[numpy.argmax(counts)]=[0,0,0]

   else:
    for l in unique:
     if l[0]==255 and l[1]==255 and l[2]==255:
      unique[l]=[0,0,0]
      counts[l]=0
   bgr=[0,0,0]
   j=0
   for k in unique:
    bgr[0] += k[0]*counts[j]
    bgr[1] += k[1]*counts[j]
    bgr[2] += k[2]*counts[j]
    j+=1
   counts=numpy.sum(counts)
   bgr[0]/=counts
   bgr[1]/=counts
   bgr[2]/=counts
   """
   bgr=[0,0,0]
   bgr[0]=unique[numpy.argmax(counts)][0]
   bgr[1]=unique[numpy.argmax(counts)][1]
   bgr[2]=unique[numpy.argmax(counts)][2]
   unique[numpy.argmax(counts)]"""
   #paste new char
   choose=random.randint(1, len(pic_array)-1)
   new_char = cv2.imread(pic_array[choose])
   cv2.imshow("result_K.png", new_char)
   cv2.waitKey()
   hsv = cv2.cvtColor(new_char, cv2.COLOR_BGR2HSV)
   lo = numpy.array([0, 0, 0])
   hi = numpy.array([255, 255, 100])  # black
   mask = cv2.inRange(hsv, lo, hi)
   new_char[mask > 0] = (bgr[0], bgr[1], bgr[2])
   new_char=cv2.resize(new_char, (seg[i][0][2], seg[i][0][3]), interpolation=cv2.INTER_CUBIC)
   cv2.imshow("result_K.png", new_char)
   cv2.waitKey()
   hsv = cv2.cvtColor(new_char, cv2.COLOR_BGR2HSV)
   lo = numpy.array([200,200,100])
   hi = numpy.array([255, 255, 255])  # black
   mask = cv2.inRange(hsv, lo, hi)
   new_char[mask > 0] = (0,0,0)
   #mask = 225 * numpy.ones(new_char.shape,new_char.dtype)
   """
   mask_2=new_char.copy()
   hsv = cv2.cvtColor(mask_2, cv2.COLOR_BGR2HSV)
   lo = numpy.array([0, 1, 0])
   hi = numpy.array([255, 255, 255])
   mask = cv2.inRange(hsv, lo, hi)
   mask_2[mask > 0] = (255,255,255)
   

   img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]]=cv2.bitwise_or(img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]], mask_2)
   cv2.imshow("result.png", img_temp_2)
   cv2.waitKey()"""
   #img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]]=cv2.addWeighted(img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]],0, new_char,1,0)

   img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]]=cv2.bitwise_and(img_temp_2[seg[i][0][1]:seg[i][0][1] + seg[i][0][3], seg[i][0][0]:seg[i][0][0] + seg[i][0][2]], new_char)

   cv2.imshow("result.png", img_temp_2)
   #normal_clone = cv2.seamlessClone(new_char, img_temp_2, mask,(seg[i][0][1]+seg[i][0][3]/2,seg[i][0][0]+seg[i][0][2]/2) , cv2.NORMAL_CLONE)
   #cv2.imshow("result.png",normal_clone)
   #img_temp_2[seg[i][0][1] :seg[i][0][1]  + seg[i][0][3], seg[i][0][0] :seg[i][0][0] + seg[i][0][2] ]
   cv2.waitKey()

   # cv2.imshow("or", char_color)
  # cv2.waitKey(0)
   #image[y:y + h, x:x + w]


#get background color
#img_temp = img_temp.reshape(-1, 3)
#print(len(img_temp))
#print(img_temp.reshape(-1, 3))
#unique, counts = numpy.unique(img_temp, axis=0, return_counts=True)
#img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[numpy.argmax(counts)]
#print(unique[numpy.argmax(counts)])
#print(counts.max())


#cv2.imshow("Output", image)
