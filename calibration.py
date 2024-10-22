import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import Helper
from helper import Line
Help = Helper()

# define world frame
# total frame 780x575 dm)

outer_left = [(50,50), (50, 575)]
outer_right = [(730, 50), (730, 575)]
mid = [(50,575), (730, 575)]
back = [(50, 50), (730, 50)]
penalty = [(189, 215), (590, 215)]
penalty_left = [(189, 50), (189, 215)]
penalty_right = [(590, 50), (590, 215)]
goal = [(299, 105), (482, 105)]
goal_left = [(299, 50), (299, 105)]
goal_right = [(482, 50), (482, 105)]

world_keypoints = [
    (50,50), (50, 575), (730, 50), (730, 575),
    (50,575), (730, 575),(50, 50), (730, 50),
    (189, 215), (590, 215), (189, 50), (189, 215),
    (590, 50), (590, 215), (299, 105), (482, 105),
    (299, 50), (299, 105), (482, 50), (482, 105)
]

# Read image
img = cv2.imread('frame.png', cv2.IMREAD_COLOR) # road.png is the filename

satAdjusted = Help.satAdjHSV(img, 5)
# plt.hist(satAdjusted.ravel(),256,[0,256]); plt.show()
mask = Help.fieldMask(satAdjusted)
masked = cv2.bitwise_and(img, img, mask=mask)

lines = list(map(Line, Help.houghLines(masked)))
linesGrouped = Help.lineGrouper(lines)

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
intersections.pop(0)
img = Help.draw_intersections(img, intersections)
matched_world_points = [(189, 215), (299, 105), (50, 50), (189, 50), (299, 50)]

H, mask = cv2.findHomography(np.asarray(matched_world_points), np.asarray(intersections))
print(H)
# Show result
# cv2.imshow("blank", cv2.cvtColor(satAdjusted, cv2.COLOR_HSV2BGR))
cv2.imshow("blank", img)
cv2.waitKey(5000)