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

known_world_points_3d = np.array([
    (50,50, 0), (50, 575, 0), (730, 50, 0), (730, 575, 0),
    (50,575, 0), (730, 575, 0),(50, 50, 0), (730, 50, 0),
    (189, 215, 0), (590, 215, 0), (189, 50, 0), (189, 215, 0),
    (590, 50, 0), (590, 215, 0), (299, 105, 0), (482, 105, 0),
    (299, 50, 0), (299, 105, 0), (482, 50, 0), (482, 105, 0)
], dtype=np.float32)

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
img = Help.draw_intersections(img, intersections)
cv2.imshow("blank", img)
cv2.waitKey(5000)

matched_world_points = np.array([
    [299, 215, 0],
    [189, 215, 0],
    [299, 105, 0],
    [50, 50, 0],
    [189, 50, 0],
    [299, 50, 0],
    [299, 50, 25]
], dtype=np.float32)

intersections.append([198, 342])
intersections= np.array(intersections, dtype=np.float32)
intersections = [intersections]
# matched_world_points = [(189, 215), (299, 105), (50, 50), (189, 50), (299, 50)]
matched_world_points = [matched_world_points]
image_size = (1080, 1920)

dist_coeffs = np.zeros((4,1))

# Initial camera matrix based on a pinhole camera model
focal_length = 1000  # You can adjust this based on approximate scene depth and image size
center_x = image_size[1] / 2
center_y = image_size[0] / 2

camera_matrix = np.array([
    [focal_length, 0, center_x],
    [0, focal_length, center_y],
    [0, 0, 1]
], dtype=np.float32)

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(matched_world_points, intersections, image_size, camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(matched_world_points, intersections, image_size, None, None)

# Print the results
print("Calibration was successful:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

# camera_matrix = np.array([[-3.04980535e+00, 3.65754478e+00, 1.04668031e+03],[3.54614115e-01, 2.71924286e-01, 2.29713728e+02],[-6.91927166e-04, -3.29538297e-04, 1.00000000e+00]])
# # H, mask = cv2.findHomography(np.asarray(matched_world_points), np.asarray(intersections))
#
# # Distortion coefficients (assuming minimal distortion)
# dist_coeffs = np.zeros((4, 1))  # Adjust if you have specific distortion coefficients
#
# # Suppose you already have an initial rotation vector and translation vector (rvec, tvec)
# # from a previous calibration or alignment
# rvec = np.zeros((3, 1))  # Initial guess for rotation
# tvec = np.zeros((3, 1))  # Initial guess for translation
#
#
# # Assume you have an initial camera matrix, rvec, tvec
# # and an image size (e.g., 1920x1080)
# image_size = (1080, 1920)
# projected_points, _ = cv2.projectPoints(known_world_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
#
# # Filter points within the image frame
# visible_points = []
# for i, pt in enumerate(projected_points.reshape(-1, 2)):
#     x, y = pt
#     if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
#         visible_points.append((pt, known_world_points_3d[i]))
#
# # Convert to array format
# visible_image_points = np.array([vp[0] for vp in visible_points], dtype=np.float32)
# visible_world_points = np.array([vp[1] for vp in visible_points], dtype=np.float32)
#
# # Apply RANSAC for robust point matching
# ransac_matches = []
# ransac_threshold = 500.0  # Distance threshold for matching points
#
# for dp in intersections:
#     # Calculate distances from the detected point to all visible projected points
#     distances = np.linalg.norm(visible_image_points - dp, axis=1)
#     min_idx = np.argmin(distances)
#     if distances[min_idx] < ransac_threshold:
#         ransac_matches.append((dp, visible_world_points[min_idx]))
#
# # Extract matched points for solvePnP
# matched_image_points = np.array([m[0] for m in ransac_matches], dtype=np.float32)
# matched_world_points = np.array([m[1] for m in ransac_matches], dtype=np.float32)
#
# # Ensure there are at least 4 points before calling solvePnP
# if len(matched_image_points) >= 4:
#     success, rvec, tvec = cv2.solvePnP(
#         matched_world_points, matched_image_points, camera_matrix, dist_coeffs,
#         rvec, tvec, useExtrinsicGuess=True
#     )
#     print(matched_world_points, matched_image_points)
#     if success:
#         print("Updated Rotation Vector (rvec):", rvec)
#         print("Updated Translation Vector (tvec):", tvec)
#     else:
#         print("Pose optimization failed.")
# else:
#     print("Not enough points to update pose.")
#
# # Show result
# # cv2.imshow("blank", cv2.cvtColor(satAdjusted, cv2.COLOR_HSV2BGR))
# cv2.imshow("blank", img)
# cv2.waitKey(5000)