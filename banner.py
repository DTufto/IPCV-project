import cv2
import numpy as np
from helper import Helper, Line

def project_banner_with_homography(image, banner, H, desired_width=200):
    h_img, w_img = image.shape[:2]
    h_banner, w_banner = banner.shape[:2]

    # Calculate center of the image
    center_x = w_img // 2
    center_y = h_img // 2

    # Create translation matrix to center the banner
    T = np.array([
        [1, 0, center_x - 500],
        [0, 1, center_y - 100],
        [0, 0, 1]
    ])

    # Calculate scaling factor to achieve desired width
    p1 = np.array([0, 0, 1])
    p2 = np.array([w_banner, 0, 1])

    # Transform points using homography
    p1_transformed = H @ p1
    p2_transformed = H @ p2

    # Convert to euclidean coordinates
    p1_transformed = p1_transformed / p1_transformed[2]
    p2_transformed = p2_transformed / p2_transformed[2]

    # Calculate current width in target space
    current_width = np.sqrt(
        (p2_transformed[0] - p1_transformed[0]) ** 2 +
        (p2_transformed[1] - p1_transformed[1]) ** 2
    )

    # Calculate scaling factor
    scale = desired_width / current_width

    # Create scaling matrix
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

    # Combine translation with original homography and scaling
    H_final = T @ H @ S

    # Warp the banner
    result = cv2.warpPerspective(banner, H_final, (w_img, h_img))

    # Create mask for blending
    mask = np.ones_like(banner)
    warped_mask = cv2.warpPerspective(mask, H_final, (w_img, h_img))

    # Blend with original image
    blended = image * (1 - warped_mask) + result * warped_mask

    return blended.astype(np.uint8)

def main():
    # Original Homography matrix
    H = np.array([
        [-3.04980535e+00, 3.65754478e+00, 1.04668031e+03],
        [3.54614115e-01, 2.71924286e-01, 2.29713728e+02],
        [-6.91927166e-04, -3.29538297e-04, 1.00000000e+00]
    ])

    # Read images
    img = cv2.imread('media/frame.png', cv2.IMREAD_COLOR)
    banner = cv2.imread('media/windows7_whopper.jpg')

    # Project banner using homography
    result = project_banner_with_homography(img, banner, H, desired_width=200)

    # Display and save result
    cv2.imshow("Result", result)
    cv2.waitKey(5000)
    cv2.imwrite('result_with_centered_banner.jpg', result)

if __name__ == "__main__":
    main()