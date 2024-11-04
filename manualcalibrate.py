import cv2
import numpy as np
from helper import Helper
from helper import Line
Help = Helper()


def calcparameters(world_points, image_points):
    world_points = [world_points]
    image_points = [image_points]
    image_size = (1080, 1920)
    dist_coeffs = np.zeros((4, 1))

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
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, image_size, camera_matrix,
                                                                        dist_coeffs,
                                                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return camera_matrix, rvecs, tvecs


world1 = np.array([
    [50, 50, 0],
    [189, 50, 0],
    [298, 50, 0],
    [353, 50, 0],
    [353, 50, 25],
    [426, 50, 0],
    [426, 50, 25],
    [298, 105, 0],
    [481, 105, 0]
], dtype=np.float32)

clip1 = np.array([
    [364, 341],
    [792, 441],
    [1212, 537],
    [1459, 595],
    [1465, 426],
    [1851, 684],
    [1856, 501],
    [937, 581],
    [1877, 828]
], dtype=np.float32)

world2 = world1

clip2 = np.array([
    [343, 330],
    [740, 449],
    [1123, 563],
    [1343, 629],
    [1342, 443],
    [1677, 733],
    [1681, 531],
    [779, 601],
    [1563, 874]
], dtype=np.float32)

world3 = np.array([
    [50, 50, 0],
    [189, 50, 0],
    [298, 50, 0],
    [353, 50, 0],
    [353, 50, 25],
    [426, 50, 0],
    [426, 50, 25],
    [298, 105, 0],
    [481, 105, 0],
    [189, 215, 0]
], dtype=np.float32)

clip3 = np.array([
    [1220, 163],
    [1340, 344],
    [1458, 523],
    [1527, 630],
    [1534, 460],
    [1641, 800],
    [1647, 614],
    [1095, 532],
    [1279, 957],
    [330, 362]
], dtype=np.float32)


camera_matrix, rvecs, tvecs = calcparameters(world1, clip1)

# Print the results
# print("Calibration was successful:", ret)
print("Camera Matrix:\n", camera_matrix)
# print("Distortion Coefficients:\n", dist_coeffs)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)