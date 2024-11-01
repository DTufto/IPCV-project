import cv2
import numpy as np
from matplotlib import pyplot as plt
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

    def __str__(self):
        return f"Point with x: {self.x} and y: {self.y}"


class Line:
    def __init__(self, line):
        x1, y1, x2, y2 = line[0]
        if y1 > y2:
            x2, y2, x1, y1 = line[0]
        self.points = [Point(x1, y1), Point(x2, y2)]
        self.angle = (self.points[0] - self.points[1]).angle()

    def __lt__(self, o):
        return self.angle < o.angle

    def __neg__(self):
        return self.points[0].x, self.points[0].y

    def __pos__(self):
        return self.points[1].x, self.points[1].y

    def __str__(self):
        return f"Line from: \n{self.points[0]} \nTo: \n{self.points[1]} \nWith angle: {self.angle}"


class Helper:
    def __init__(self):
        self.last_valid_mask = None
        self.green_ranges = []
        return

    def dilate(self, img, iters=1):
        return cv2.dilate(img, kernel, iterations=iters)

    def erode(self, img, iters=1):
        return cv2.erode(img, kernel, iterations=iters)

    def analyze_image_stats(self, img):
        """Analyze color statistics of the image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Calculate histogram for hue channel
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])

        # Find dominant hue peaks
        peaks = np.argpartition(hist_h.flatten(), -3)[-3:]  # Top 3 hue values

        # Calculate statistics
        avg_brightness = np.mean(v)
        avg_saturation = np.mean(s)
        std_h = np.std(h)
        std_s = np.std(s)

        return {
            'dominant_hues': peaks,
            'avg_brightness': avg_brightness,
            'avg_saturation': avg_saturation,
            'hue_std': std_h,
            'sat_std': std_s
        }

    def remove_small_regions(self, mask, min_size=1000):
        """Remove small disconnected regions from the mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Find the largest component (excluding background)
        sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        if len(sizes) == 0:
            return mask

        max_size = np.max(sizes)

        # Create new mask with only large regions
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size or stats[i, cv2.CC_STAT_AREA] >= max_size * 0.1:
                clean_mask[labels == i] = 255

        return clean_mask

    def detect_players(self, img, hsv):
        """
        Detect players using multiple techniques including jersey colors and local variance
        """
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

    def create_grass_mask(self, hsv):
        """
        Create a robust grass mask that accounts for color variations
        """
        h, s, v = cv2.split(hsv)

        # Multiple grass color ranges
        grass_masks = []

        # Main grass color
        grass_masks.append(cv2.inRange(hsv, np.array([35, 40, 30]), np.array([85, 255, 200])))

        # Lighter grass / chalk lines
        grass_masks.append(cv2.inRange(hsv, np.array([35, 20, 150]), np.array([85, 100, 255])))

        # Darker grass shadows
        grass_masks.append(cv2.inRange(hsv, np.array([35, 40, 20]), np.array([85, 255, 150])))

        # Combine all grass masks
        combined_mask = grass_masks[0]
        for mask in grass_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def create_field_mask(self, img):
        """
        Create an improved field mask with better handling of players and field variations
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Create initial field mask
        field_mask = self.create_grass_mask(hsv)

        # Initial morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
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

    def houghLines(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        ret, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        edges = cv2.Canny(gray, 120, 240)
        # Detect points that form a line
        lines = cv2.HoughLinesP(gray, 3, np.pi / 720, 1000, minLineLength=200, maxLineGap=40)
        return lines

    def satAdjHSV(self, imgBGR, sAdj):
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imgHSV.astype('int'))
        s = s * sAdj
        s = np.clip(s, 0, 255)
        return cv2.merge([h, s, v]).astype('uint8')

    def thresholdBand(self, img, lower, upper):
        _, mask0 = cv2.threshold(img, lower, 255, cv2.THRESH_BINARY)
        _, mask1 = cv2.threshold(img, upper, 255, cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(mask0, mask0, mask=mask1)

    def fieldMask(self, imgHSV):
        """Legacy field mask method - kept for compatibility"""
        (h, s, v) = cv2.split(imgHSV)
        mask = self.thresholdBand(h, 40, 70)
        mask = self.erode(mask, 14)
        mask = self.dilate(mask, 50)
        return mask

    def lineGrouper(self, lines):
        """Improved line grouping that better handles parallel lines."""
        if not lines:
            return []

        # Sort lines by angle
        lines.sort()

        # Initialize groups
        groups = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            current_line = lines[i]
            reference_line = current_group[0]  # Use first line in group as reference

            # Check if lines are parallel (within tolerance)
            angle_diff = abs(current_line.angle - reference_line.angle)
            if angle_diff < 10 or angle_diff > 170:
                # Lines are parallel - check if they're close enough
                if self.are_lines_close(current_line, reference_line):
                    current_group.append(current_line)
                else:
                    # Start new group if lines are far apart
                    if len(current_group) > 0:
                        groups.append(current_group)
                    current_group = [current_line]
            else:
                # Different angle - start new group
                if len(current_group) > 0:
                    groups.append(current_group)
                current_group = [current_line]

        # Add last group
        if len(current_group) > 0:
            groups.append(current_group)

        return groups

    def get_line_intersection(self, line1, line2):
        x1, y1 = -line1
        x2, y2 = +line1
        x3, y3 = -line2
        x4, y4 = +line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-8:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        if not (0 <= t <= 1):
            return None

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (int(x), int(y))

    def are_lines_close(self, line1, line2, max_distance=50):
        """Check if two parallel lines are close to each other."""
        # Get midpoints of lines
        mid1 = ((line1.points[0].x + line1.points[1].x) / 2,
                (line1.points[0].y + line1.points[1].y) / 2)
        mid2 = ((line2.points[0].x + line2.points[1].x) / 2,
                (line2.points[0].y + line2.points[1].y) / 2)

        # Calculate distance between midpoints
        distance = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)

        return distance < max_distance

    def find_field_intersections(self, linesGrouped, mask, min_distance=40):
        """Find intersections between roughly perpendicular lines."""
        intersections = []

        for i in range(len(linesGrouped)):
            for j in range(i + 1, len(linesGrouped)):
                # Get representative lines from each group
                line1 = linesGrouped[i][0]  # Use first line as representative
                line2 = linesGrouped[j][0]

                # Check if lines are roughly perpendicular (90° ± 20°)
                angle_diff = abs(abs(line1.angle - line2.angle) - 90)
                if angle_diff > 20:
                    continue

                # Find intersection
                intersection = self.get_line_intersection(line1, line2)

                if intersection is None:
                    continue

                x, y = intersection

                # Check if intersection is within mask and not too close to existing points
                if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and
                        mask[y, x] > 0):

                    is_unique = True
                    for existing_x, existing_y in intersections:
                        dist = np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
                        if dist < min_distance:
                            is_unique = False
                            break

                    if is_unique:
                        intersections.append((x, y))

        return intersections

    def draw_intersections(self, img, intersections, radius=5, color=(0, 255, 0), thickness=-1):
        result = img.copy()
        for x, y in intersections:
            cv2.circle(result, (x, y), radius, color, thickness)
        return result

    def detect_lines_on_field(self, frame, field_mask):
        """Detect lines on the field using optimized preprocessing within masked region"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find bounding box of the field mask
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
        threshold_offset = 30  # This can be adjusted
        _, roi_thresh = cv2.threshold(
            roi_masked,
            mean_val + threshold_offset,
            255,
            cv2.THRESH_BINARY
        )

        # Optional: Quick noise reduction using a small gaussian blur
        roi_thresh = cv2.GaussianBlur(roi_thresh, (3, 3), 0)

        # Edge enhancement using Sobel operators (faster than Canny)
        sobelx = cv2.Sobel(roi_thresh, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_thresh, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalize gradient to 0-255 range
        gradient = np.uint8(gradient * 255 / gradient.max())

        # Simple threshold on gradient image
        _, edges = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)

        # Use probabilistic Hough transform with optimized parameters
        min_line_length = 50
        max_line_gap = 30
        threshold = 50

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,  # Reduced angle resolution for speed
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # If no lines detected, try with more sensitive parameters
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

                # Filter based on angle
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if (angle < 20 or (90 - 20) <= angle <= (90 + 20) or angle > 160):
                    filtered_lines.append(np.array([[x1, y1, x2, y2]]))

            lines = np.array(filtered_lines) if filtered_lines else None

        return lines