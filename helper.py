import cv2
import numpy as np
import math

kernel = np.ones((3, 3), np.uint8)


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def angle(self):
        if self.x == 0:
            return 90.0
        return math.degrees(math.atan(self.y / self.x))

    def __str__(self):
        return f"Vector with x: {self.x} and y: {self.y}"


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y)

class Line:
    def __init__(self, line):
        x1, y1, x2, y2 = line[0]
        self.points = [Point(x1, y1), Point(x2, y2)]
        self.angle = (self.points[0] - self.points[1]).angle()

    def __lt__(self, o):
        return self.angle < o.angle

    def __neg__(self):
        return self.points[0].x, self.points[0].y

    def __pos__(self):
        return self.points[1].x, self.points[1].y

class Helper:
    def __init__(self):
        self.last_valid_mask = None

    def detect_players(self, img, hsv):
        """Detect players using jersey colors and local variance"""
        h, s, v = cv2.split(hsv)

        # Jersey color detection
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        # Local variance for detailed areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, variance_mask = cv2.threshold(
            cv2.GaussianBlur(cv2.Laplacian(gray, cv2.CV_8U), (5, 5), 0),
            10, 255, cv2.THRESH_BINARY
        )

        # Combine all player detection methods
        player_mask = cv2.bitwise_or(blue_mask, red_mask)
        player_mask = cv2.bitwise_or(player_mask, black_mask)
        player_mask = cv2.bitwise_or(player_mask, variance_mask)

        # Clean up player mask
        kernel = np.ones((5, 5), np.uint8)
        player_mask = cv2.dilate(player_mask, kernel, iterations=2)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)

        return player_mask

    def create_grass_mask(self, hsv, debug_folder=None):
        """
        Create a grass mask that accounts for color variations and handles LED boards
        """
        # Multiple grass color ranges
        grass_masks = []

        # Main grass color - tightened range to avoid LED interference
        grass_masks.append(cv2.inRange(hsv, np.array([35, 50, 30]), np.array([85, 255, 200])))

        # Lighter grass / chalk lines - adjusted saturation
        grass_masks.append(cv2.inRange(hsv, np.array([35, 30, 150]), np.array([85, 90, 255])))

        # Darker grass shadows
        grass_masks.append(cv2.inRange(hsv, np.array([35, 50, 20]), np.array([85, 255, 150])))

        # Combine all grass masks
        combined_mask = grass_masks[0]
        for mask in grass_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Additional LED board filtering
        def filter_led_boards(mask, hsv):
            # Detect super bright and saturated areas (typical of LED boards)
            led_mask = cv2.inRange(hsv, np.array([0, 200, 200]), np.array([180, 255, 255]))

            # Remove thin horizontal strips that are likely LED boards
            kernel_horizontal = np.ones((1, 15), np.uint8)
            led_lines = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_horizontal)

            # Remove detected LED regions from the grass mask
            return cv2.bitwise_and(mask, cv2.bitwise_not(led_lines))

        # Apply LED filtering
        combined_mask = filter_led_boards(combined_mask, hsv)

        # Additional post-processing to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    def create_field_mask(self, img):
        """Create an improved field mask with better handling of players and field variations"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create initial field mask
        field_mask = self.create_grass_mask(hsv)

        # Initial morphological operations
        kernel_medium = np.ones((7, 7), np.uint8)
        kernel_large = np.ones((21, 21), np.uint8)

        # Fill small gaps (like lines)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_medium)

        # Find the main field area using contours
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (the field)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create mask from largest contour
            field_mask = np.zeros_like(field_mask)
            cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)

            # Fill any holes in the field mask
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)

        # Detect and remove players
        player_mask = self.detect_players(img, hsv)
        field_mask[player_mask > 0] = 0

        # Fill gaps created by player removal
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)

        # Store last valid mask in case we need it
        if np.sum(field_mask) > 0:
            self.last_valid_mask = field_mask.copy()
        elif self.last_valid_mask is not None:
            field_mask = self.last_valid_mask.copy()

        return field_mask

    def lineGrouper(self, lines):
        """Line grouping using angle and perpendicular distance."""
        if not lines:
            return []

        # Sort lines by angle
        lines.sort()

        # Initialize groups
        groups = []
        used_lines = set()

        for i, line in enumerate(lines):
            if i in used_lines:
                continue

            current_group = [line]
            used_lines.add(i)

            # Compare with remaining lines
            for j in range(i + 1, len(lines)):
                if j in used_lines:
                    continue

                test_line = lines[j]
                angle_diff = abs(test_line.angle - line.angle)

                # Check if lines are parallel (within tolerance)
                if angle_diff < 5 or angle_diff > 175:
                    # Check perpendicular distance
                    if self.are_lines_close(line, test_line):
                        current_group.append(test_line)
                        used_lines.add(j)

            if len(current_group) > 0:
                # Average the lines in the group
                groups.append(self.merge_group_lines(current_group))

        return groups

    def merge_group_lines(self, group):
        """Merge lines in a group by taking the furthest endpoints."""
        if len(group) == 1:
            return group

        # Initialize with first line's endpoints
        points = []
        for line in group:
            points.append((line.points[0].x, line.points[0].y))  # Start point
            points.append((line.points[1].x, line.points[1].y))  # End point

        # Convert to numpy array for easier manipulation
        points = np.array(points)

        # Get the primary direction of the line group (using first line as reference)
        ref_line = group[0]
        dx = ref_line.points[1].x - ref_line.points[0].x
        dy = ref_line.points[1].y - ref_line.points[0].y

        # Calculate the angle of the line
        angle = math.atan2(dy, dx)

        # Create rotation matrix to align line with x-axis
        rotation_matrix = np.array([
            [math.cos(-angle), -math.sin(-angle)],
            [math.sin(-angle), math.cos(-angle)]
        ])

        # Rotate all points
        rotated_points = np.dot(points, rotation_matrix.T)

        # Find extremes in rotated space (now aligned with x-axis)
        min_x_idx = np.argmin(rotated_points[:, 0])
        max_x_idx = np.argmax(rotated_points[:, 0])

        # Get the original points that correspond to these extremes
        start_point = points[min_x_idx]
        end_point = points[max_x_idx]

        # Create new merged line
        merged_line = [np.array([[int(start_point[0]), int(start_point[1]),
                                  int(end_point[0]), int(end_point[1])]])]
        return [Line(merged_line[0])]

    def are_lines_close(self, line1, line2, max_distance=30):
        """Check if two parallel lines are close to each other using perpendicular distance."""
        # Get line vectors
        vec1 = (line1.points[1].x - line1.points[0].x, line1.points[1].y - line1.points[0].y)
        length1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)

        # Normalize vector
        if length1 > 0:
            vec1 = (vec1[0] / length1, vec1[1] / length1)

        # Calculate perpendicular vector
        perp_vec = (-vec1[1], vec1[0])

        # Get midpoints
        mid1 = ((line1.points[0].x + line1.points[1].x) / 2,
                (line1.points[0].y + line1.points[1].y) / 2)
        mid2 = ((line2.points[0].x + line2.points[1].x) / 2,
                (line2.points[0].y + line2.points[1].y) / 2)

        # Calculate perpendicular distance
        diff_x = mid2[0] - mid1[0]
        diff_y = mid2[1] - mid1[1]
        perp_dist = abs(diff_x * perp_vec[0] + diff_y * perp_vec[1])

        return perp_dist < max_distance

    def get_line_intersection(self, line1, line2):
        """
        Find intersection point of two lines using their endpoints.
        Returns None if lines are parallel, don't intersect, or are too far apart.
        """
        x1, y1 = -line1
        x2, y2 = +line1
        x3, y3 = -line2
        x4, y4 = +line2

        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-8:  # Lines are parallel
            return None

        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Calculate intersection point
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        # Check if intersection occurs within or very close to both line segments
        margin = 20  # Allow for some margin beyond line endpoints

        # Check if intersection is within the bounding box of each line (with margin)
        in_bounds_1 = (
                min(x1, x2) - margin <= x <= max(x1, x2) + margin and
                min(y1, y2) - margin <= y <= max(y1, y2) + margin
        )

        in_bounds_2 = (
                min(x3, x4) - margin <= x <= max(x3, x4) + margin and
                min(y3, y4) - margin <= y <= max(y3, y4) + margin
        )

        if not (in_bounds_1 and in_bounds_2):
            return None

        return (int(x), int(y))

    def find_field_intersections(self, linesGrouped, mask, min_distance=30):
        """Find valid intersection points with improved filtering."""
        intersections = []

        # Compare each group with every other group
        for i in range(len(linesGrouped)):
            for j in range(i + 1, len(linesGrouped)):
                # Get angle between groups
                angle_diff = abs(linesGrouped[i][0].angle - linesGrouped[j][0].angle)

                # Skip nearly parallel lines
                if angle_diff < 20 or angle_diff > 160:
                    continue

                # Compare each line pair
                for line1 in linesGrouped[i]:
                    for line2 in linesGrouped[j]:
                        intersection = self.get_line_intersection(line1, line2)

                        if intersection is None:
                            continue

                        x, y = intersection

                        # Check if point is within image bounds and field mask
                        if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and
                                mask[y, x] > 0):

                            # Check distance from existing intersections
                            is_unique = True
                            for existing_x, existing_y in intersections:
                                dist = math.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
                                if dist < min_distance:
                                    is_unique = False
                                    break

                            if is_unique:
                                intersections.append((x, y))

        return intersections

    def draw_intersections(self, img, intersections, radius=5, color=(0, 255, 0), thickness=-1):
        """
        Draw the intersection points on the image
        """
        result = img.copy()
        for x, y in intersections:
            cv2.circle(result, (x, y), radius, color, thickness)
        return result



    def detect_lines_on_field(self, frame, field_mask):
        """Detect lines on the field using Canny edge detection within masked region"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if field_mask is None:
            return None

        # Find non-zero points in mask
        nz = cv2.findNonZero(field_mask)
        if nz is None:
            return None

        # Get bounding rectangle of mask
        x, y, w, h = cv2.boundingRect(nz)

        # Extract ROI from grayscale image and mask
        roi_gray = gray[y:y + h, x:x + w]
        roi_mask = field_mask[y:y + h, x:x + w]

        # Apply mask to ROI
        roi_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)

        # Calculate mean brightness of the masked area for dynamic thresholding
        mean_val = cv2.mean(roi_masked, roi_mask)[0]

        # Dynamic thresholding based on mean brightness
        threshold_offset = 30
        _, roi_thresh = cv2.threshold(
            roi_masked,
            mean_val + threshold_offset,
            255,
            cv2.THRESH_BINARY
        )

        # Noise reduction using small gaussian blur
        roi_thresh = cv2.GaussianBlur(roi_thresh, (3, 3), 0)

        # Use Canny edge detection instead of Sobel
        # Lower threshold set to 50, upper threshold to 150
        edges = cv2.Canny(roi_thresh, 30, 150)

        # Use probabilistic Hough transform
        min_line_length = 50
        max_line_gap = 70
        threshold = 50

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # Try with more sensitive parameters if no lines detected
        if lines is None or len(lines) < 2:
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=30,
                maxLineGap=40
            )

        # Adjust line coordinates back to original image space
        if lines is not None:
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Adjust coordinates back to original image space
                x1, y1 = x1 + x, y1 + y
                x2, y2 = x2 + x, y2 + y

                filtered_lines.append(np.array([[x1, y1, x2, y2]]))

            lines = np.array(filtered_lines) if filtered_lines else None

        return lines