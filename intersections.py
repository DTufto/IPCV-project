import cv2
import numpy as np


def detect_line_intersections(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        if np.pi / 4 < theta < 3 * np.pi / 4:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    # Find intersections
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            rho1, theta1 = h_line[0]
            rho2, theta2 = v_line[0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append((x0, y0))

    # Draw intersections on the image
    for point in intersections:
        cv2.circle(img, point, 5, (0, 0, 255), -1)

    # Save the result
    cv2.imwrite('result_intersections.jpg', img)

    return intersections


# Usage
image_path = 'media/corner-still.jpeg'
intersections = detect_line_intersections(image_path)
print(f"Found {len(intersections)} intersections")