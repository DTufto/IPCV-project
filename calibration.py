import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('veld.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Use canny edge detection
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
# # Apply HoughLinesP method to
# # to directly obtain line end points
# lines_list = []
# lines = cv2.HoughLinesP(
#     edges,  # Input edge image
#     1,  # Distance resolution in pixels
#     np.pi / 180,  # Angle resolution in radians
#     threshold=100,  # Min number of votes for valid line
#     minLineLength=5,  # Min allowed length of line
#     maxLineGap=10  # Max allowed gap between line for joining them
# )
#
# # Iterate over points
# for points in lines:
#     # Extracted points nested in the list
#     x1, y1, x2, y2 = points[0]
#     # Draw the lines joing the points
#     # On the original image
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     # Maintain a simples lookup list for points
#     lines_list.append([(x1, y1), (x2, y2)])

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image (for visualization)
sift_image = cv2.drawKeypoints(image, keypoints, None)

# Show the image with keypoints
cv2.imshow('SIFT Keypoints', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow('detectedLines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

# # Plot a line given two points
# def plot_line(points, label):
#     x_values = [points[0][0], points[1][0]]
#     y_values = [points[0][1], points[1][1]]
#     plt.plot(x_values, y_values, label=label)
#
# # Create a plot
# plt.figure
#
# # Plot each line on the field
# plot_line(outer_left, 'Outer Left')
# plot_line(outer_right, 'Outer Right')
# plot_line(mid, 'Mid Line')
# plot_line(back, 'Back Line')
# plot_line(penalty, 'Penalty Line')
# plot_line(penalty_left, 'Penalty Left')
# plot_line(penalty_right, 'Penalty Right')
# plot_line(goal, 'Goal Line')
# plot_line(goal_left, 'Goal Left')
# plot_line(goal_right, 'Goal Right')
#
# # Customize the plot
# plt.xlim(0, 780)  # Adjust based on expected field size
# plt.ylim(0, 575)  # Adjust based on expected field size
# plt.gca().invert_yaxis()  # Invert Y axis to match typical soccer field layout (origin at top-left)
# plt.xlabel("X coordinate (in dm)")
# plt.ylabel("Y coordinate (in dm)")
# plt.title("Soccer Field Lines")
# plt.legend(loc="lower right")
#
# # Show the plot
# plt.grid(True)
# plt.show()