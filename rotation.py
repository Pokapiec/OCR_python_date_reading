import numpy as np
import cv2
import math
from scipy import ndimage
import imutils

img = cv2.imread('images/im5.jpg')

scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


cv2.imshow("Before", resized)    
key = cv2.waitKey(0)

img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []


for [[x1, y1, x2, y2]] in lines:
    cv2.line(resized, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

cv2.imshow("Detected lines", resized)    
key = cv2.waitKey(0)

median_angle = np.median(angles)
img_rotated = ndimage.rotate(resized, median_angle)

print(f"Angle is {median_angle:.04f}")
#cv2.imwrite('rotated.jpg', img_rotated)  

rotated_image = imutils.rotate(resized, median_angle)
cv2.imshow("After", rotated_image)
key = cv2.waitKey(0)
