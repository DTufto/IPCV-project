import cv2
import numpy as np
from helper import Helper, Line


def project_banner_from_intersections(image, banner, intersections, angle_degrees=150):
    """
    Project a banner onto an image using intersection points and a specified angle.

    Args:
        image: Source image to project onto
        banner: Banner image to project
        intersections: List of intersection points
        angle_degrees: Angle of the banner in degrees (0 is vertical, 90 is perpendicular to the line)
    """
    if len(intersections) < 2:
        raise ValueError("Need at least 2 intersection points to place banner")

    # Sort intersections by x-coordinate
    sorted_intersections = sorted(intersections, key=lambda p: p[0])
    left_point = sorted_intersections[0]
    right_point = sorted_intersections[1]

    h, w = image.shape[:2]

    # Calculate banner width based on intersection points
    banner_width = np.sqrt((right_point[0] - left_point[0]) ** 2 +
                           (right_point[1] - left_point[1]) ** 2)

    # Set banner height proportional to width while maintaining aspect ratio
    banner_height = banner_width * 0.3

    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    direction_length = np.sqrt(dx * dx + dy * dy)

    dx = dx / direction_length
    dy = dy / direction_length

    perp_dx = -dy
    perp_dy = dx

    angle_radians = np.deg2rad(angle_degrees)
    proj_dx = perp_dx * np.cos(angle_radians) - dx * np.sin(angle_radians)
    proj_dy = perp_dy * np.cos(angle_radians) - dy * np.sin(angle_radians)

    # Create destination points for the banner
    banner_pts_2d = np.array([
        [left_point[0], left_point[1]],  # Bottom left
        [right_point[0], right_point[1]],  # Bottom right
        # Top right: using projection vector
        [right_point[0] + proj_dx * banner_height,
         right_point[1] + proj_dy * banner_height],
        # Top left: using projection vector
        [left_point[0] + proj_dx * banner_height,
         left_point[1] + proj_dy * banner_height]
    ], dtype=np.float32)

    # Source points from banner image
    h_banner, w_banner = banner.shape[:2]
    src_pts = np.array([
        [0, h_banner - 1],  # Bottom left
        [w_banner - 1, h_banner - 1],  # Bottom right
        [w_banner - 1, 0],  # Top right
        [0, 0]  # Top left
    ], dtype=np.float32)

    # Calculate perspective transform and warp banner
    M = cv2.getPerspectiveTransform(src_pts, banner_pts_2d)
    result = cv2.warpPerspective(banner, M, (w, h))
    mask = cv2.warpPerspective(np.ones_like(banner), M, (w, h))

    # Blend with original image
    blended = image * (1 - mask) + result * mask

    return blended.astype(np.uint8)


# Main script
def main():
    img = cv2.imread('media/corner-still.jpeg', cv2.IMREAD_COLOR)
    banner = cv2.imread('media/windows7_whopper.jpg')
    Help = Helper()
    satAdjusted = Help.satAdjHSV(img, 5)
    mask = Help.fieldMask(satAdjusted)
    masked = cv2.bitwise_and(img, img, mask=mask)

    lines = list(map(Line, Help.houghLines(masked)))
    linesGrouped = Help.lineGrouper(lines)
    intersections = Help.find_field_intersections(linesGrouped, mask)

    for x, y in intersections:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Project banner using intersections
    result = project_banner_from_intersections(img, banner, intersections)
    # Display and save result
    cv2.imshow("Result", result)
    cv2.waitKey(5000)
    cv2.imwrite('result_with_banner.jpg', result)


if __name__ == "__main__":
    main()