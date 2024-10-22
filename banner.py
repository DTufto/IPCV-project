import cv2
import numpy as np

def project_angled_banner_with_camera(image, banner, K, banner_rect, angle_degrees=60):
    h, w = image.shape[:2]

    # Rotation and translation vector
    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 10.0])  # Assuming camera is 10 units away from the field center

    # Banner in 3d
    x, y, width, height = banner_rect
    banner_pts_3d = np.array([
        [x, y, 0],
        [x + width, y, 0],
        [x + width, y + height, 0],
        [x, y + height, 0]
    ], dtype=np.float32)

    # Rotate banner points
    center = banner_pts_3d.mean(axis=0)
    rotation_matrix = cv2.Rodrigues(np.array([0, np.radians(angle_degrees), 0]))[0]
    banner_pts_3d_rotated = np.dot(banner_pts_3d - center, rotation_matrix.T) + center

    # Project rotated 3D banner points to 2D image points
    banner_pts_2d, _ = cv2.projectPoints(banner_pts_3d_rotated, rvec, tvec, K, None)
    banner_pts_2d = banner_pts_2d.reshape(-1, 2).astype(np.float32)

    # Warp the banner image
    h_banner, w_banner = banner.shape[:2]
    src_pts_banner = np.array([[0, 0], [w_banner-1, 0], [w_banner-1, h_banner-1], [0, h_banner-1]], dtype=np.float32)
    M_banner = cv2.getPerspectiveTransform(src_pts_banner, banner_pts_2d)
    result = cv2.warpPerspective(banner, M_banner, (w, h))
    mask = cv2.warpPerspective(np.ones_like(banner), M_banner, (w, h))

    # Attempt at shadow
    shadow_offset = np.array([0.1, 0.1, 0])  # Offset in 3D space
    shadow_pts_3d = banner_pts_3d_rotated + shadow_offset
    shadow_pts_2d, _ = cv2.projectPoints(shadow_pts_3d, rvec, tvec, K, None)
    shadow_pts_2d = shadow_pts_2d.reshape(-1, 2).astype(np.float32)
    M_shadow = cv2.getPerspectiveTransform(src_pts_banner, shadow_pts_2d)
    shadow = cv2.warpPerspective(np.ones_like(banner) * 0.3, M_shadow, (w, h))

    # Blend shadow, then banner with the original image
    blended = image * (1 - shadow) + shadow * image
    blended = blended * (1 - mask) + result

    return blended.astype(np.uint8)

image = cv2.imread('media/corner-still.jpeg')
banner = cv2.imread('media/windows7_whopper.jpg')

# Random intrinsinc parameters
K = np.array([
    [1000, 0, image.shape[1]/2],
    [0, 1000, image.shape[0]/2],
    [0, 0, 1]
])

# Where is the field?
field_corners_3d = np.array([
    [-50, -25, 0],
    [50, -25, 0],
    [50, 25, 0],
    [-50, 25, 0]
], dtype=np.float32)

# Hardcoded banner positions for now
banner_width = 30
banner_height = 5
banner_rect = (-banner_width/2, -5, banner_width, banner_height)

result = project_angled_banner_with_camera(image, banner, K, banner_rect, angle_degrees=60)

cv2.imwrite('result.jpg', result)
