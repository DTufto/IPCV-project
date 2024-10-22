import cv2
import numpy as np

def project_angled_banner_with_camera(image, banner, K, field_corners_3d, banner_rect, angle_degrees=60):
    h, w = image.shape[:2]

    # Define rotation vector (assuming camera is looking straight at the field)
    rvec = np.array([0.0, 0.0, 0.0])

    # Define translation vector (adjust these values based on your setup)
    tvec = np.array([0.0, 0.0, 10.0])  # Assuming camera is 10 units away from the field center

    # Project 3D field corners to 2D image points
    field_corners_2d, _ = cv2.projectPoints(field_corners_3d, rvec, tvec, K, None)
    field_corners_2d = field_corners_2d.reshape(-1, 2)

    # Define banner points in 3D space
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

    # Create shadow (simplified)
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

# Example usage
image = cv2.imread('media/corner-still.jpeg')
banner = cv2.imread('media/Pauline.jpg')

# Define intrinsic camera matrix (example values, adjust as needed)
K = np.array([
    [1000, 0, image.shape[1]/2],
    [0, 1000, image.shape[0]/2],
    [0, 0, 1]
])

# Define the corners of the field in 3D space (assuming field is 100x50 meters)
field_corners_3d = np.array([
    [-50, -25, 0],  # Top-left corner of the field
    [50, -25, 0],   # Top-right corner of the field
    [50, 25, 0],    # Bottom-right corner of the field
    [-50, 25, 0]    # Bottom-left corner of the field
], dtype=np.float32)

# Define the banner position and size in the 3D field space
# Centered banner: x = 0 (center of field), y slightly above center
banner_width = 30  # meters
banner_height = 5  # meters
banner_rect = (-banner_width/2, -5, banner_width, banner_height)  # x, y, width, height in meters

result = project_angled_banner_with_camera(image, banner, K, field_corners_3d, banner_rect, angle_degrees=60)

cv2.imwrite('result.jpg', result)
