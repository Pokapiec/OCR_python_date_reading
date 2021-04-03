import re
import cv2
import pytesseract
from pytesseract import Output

img = cv2.imread(r'images/im1.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


#otrzymywanie samych liczb/cyfr
#custom_config = r'--oem 3 --psm 6 outputbase digits'
#print(pytesseract.image_to_string(img, config=custom_config))

date_pattern = '^[0-9]*$'


n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        if re.match(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)



custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(img, config=custom_config))

cv2.imshow('img', img)
cv2.waitKey(0)