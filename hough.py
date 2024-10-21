import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper import helper
Help = helper()

# Read image 
img = cv2.imread('frame.png', cv2.IMREAD_COLOR) # road.png is the filename

satAdjusted = Help.satAdjHSV(img, 5)
# plt.hist(satAdjusted.ravel(),256,[0,256]); plt.show()
mask = Help.fieldMask(satAdjusted)
masked = cv2.bitwise_and(img, img, mask=mask)

lines = Help.houghLines(masked)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
# cv2.imshow("blank", cv2.cvtColor(satAdjusted, cv2.COLOR_HSV2BGR))
cv2.imshow("blank", img)
cv2.waitKey(5000)