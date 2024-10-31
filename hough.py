import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper import Helper
from helper import Line
import colorsys

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

Help = Helper()

# Read image 
img = cv2.imread('frame.png', cv2.IMREAD_COLOR) # road.png is the filename

satAdjusted = Help.satAdjHSV(img.copy(), 5)
# plt.hist(satAdjusted.ravel(),256,[0,256]); plt.show()
mask = Help.fieldMask(satAdjusted)
masked = cv2.bitwise_and(img, img, mask=mask)

linesP = list(map(Line, Help.houghPalen(img)))
linesFilt = Help.lineFilter(linesP, 300, 40, 80, 100)
lines = list(map(Line, Help.houghLines(masked)))
linesGrouped = Help.lineGrouper(lines + linesFilt)
linesFiltered = Help.lineJoiner(linesGrouped.copy())
# for group in linesGrouped:
#     print(len(group))
#     print("sep")
# Draw lines on the image
for index, group in enumerate(linesFiltered):
    for line in group:
        # print(line.angle/180)
        color = hsv2rgb(index/4, 0.5, 1)
        cv2.line(img, -line, +line, color, 3)
        cv2.circle(img, -line, radius=10, color=(0, 0, 255), thickness=-1)

# for line in linesFilt:
#     color = hsv2rgb(line.angle/180, 0.5, 1)
#     cv2.line(img, -line, +line, color, 3)
#     cv2.circle(img, -line, radius=10, color=(0, 0, 255), thickness=-1)
# print(len(linesGrouped))
# Show result
# cv2.imshow("blank", cv2.cvtColor(satAdjusted, cv2.COLOR_HSV2BGR))
cv2.imshow("blank", img)
cv2.waitKey(50000)
