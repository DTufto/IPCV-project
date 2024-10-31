import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper import Helper
from helper import Line
Help = Helper()

# Read image 
img = cv2.imread('media/frame.png', cv2.IMREAD_COLOR) # road.png is the filename

satAdjusted = Help.satAdjHSV(img, 5)
# plt.hist(satAdjusted.ravel(),256,[0,256]); plt.show()
mask = Help.fieldMask(satAdjusted)
masked = cv2.bitwise_and(img, img, mask=mask)

lines = list(map(Line, Help.houghLines(masked)))
linesGrouped = Help.lineGrouper(lines)
for group in linesGrouped:
    print(len(group))
    print("sep")
# Draw lines on the image
for index, group in enumerate(linesGrouped):
    color = (255, 0, 255)
    if index == 0:
        color = (255, 0, 0)
    elif index == 1:
        color = (0, 0, 255)
    for line in group:
        cv2.line(img, -line, +line, color, 3)

intersections = Help.find_field_intersections(linesGrouped, mask)
img = Help.draw_intersections(img, intersections)
# Show result
# cv2.imshow("blank", cv2.cvtColor(satAdjusted, cv2.COLOR_HSV2BGR))
cv2.imshow("blank", img)
cv2.waitKey(5000)