import cv2
import numpy as np
import math
from enum import Enum, auto


class PointType(Enum):
    """Enum for different types of points on the football field"""
    EMPTY = auto()
    LCORNER = auto()         # Left corner
    LGOALLINEX16M = auto()   # Left 16.5m intersection with goal line
    LGOALLINEX5M = auto()    # Left 5.5m intersection with goal line
    LM16 = auto()           # Left 16.5m box corner
    LM5 = auto()            # Left 5.5m box corner
    LGOALPOST = auto()      # Left goal post
    RGOALPOST = auto()      # Right goal post
    RM5 = auto()            # Right 5.5m box corner
    RM16 = auto()           # Right 16.5m box corner
    RGOALLINEX5M = auto()   # Right 5.5m intersection with goal line
    RGOALLINEX16M = auto()  # Right 16.5m intersection with goal line
    RCORNER = auto()        # Right corner


class LineType(Enum):
    """Enum for different types of lines on the football field"""
    EMPTY = auto()
    GOALLINE = auto()        # Goal line
    LEFTSIDELINE = auto()    # Left sideline
    RIGHTSIDELINE = auto()   # Right sideline
    LINE16L = auto()         # Left 16.5m line
    LINE16 = auto()          # Central 16.5m line
    LINE16R = auto()         # Right 16.5m line
    LINE5L = auto()          # Left 5.5m line
    LINE5 = auto()           # Central 5.5m line
    LINE5R = auto()          # Right 5.5m line


class Vector:
    """Class representing a 3D vector"""
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def angle(self) -> float:
        """Calculate angle in degrees relative to x-axis"""
        if self.x == 0:
            return 90.0
        return math.degrees(math.atan(self.y / self.x))

    def mag(self) -> float:
        """Calculate magnitude of vector"""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __str__(self) -> str:
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

    def __mul__(self, other: 'Vector') -> float:  # Dot product
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y, -self.z)

    def __le__(self, other: float) -> bool:
        return self.mag() <= other

    def __and__(self, other: 'Vector') -> 'Vector':  # Cross product
        return Vector(
            self.y * other.z - self.z * other.y,
            -1 * (self.x * other.z - self.z * other.x),
            self.x * other.y - self.y * other.x
        )


class Point:
    """Class representing a point with line intersections"""
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = np.int64(x)
        self.y = np.int64(y)
        self.z = np.int64(z)
        self.interLines = []
        self.name = PointType.EMPTY

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point') -> Vector:
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __str__(self) -> str:
        return f"Point(name={self.name}, x={self.x}, y={self.y}, z={self.z})"

    def onLine(self, line: 'Line', epsilon: float = 10000) -> bool:
        """Check if point lies on a line within epsilon tolerance"""
        vec0 = line.points[0] - line.points[1]
        vec1 = line.points[1] - self
        crossResult = (vec0 & vec1)
        return crossResult <= epsilon

    def withinLineSquare(self, line: 'Line', bounds: float = 10) -> bool:
        """Check if point lies within bounded rectangle around line"""
        maxY = max(line.points[0].y, line.points[1].y)
        minY = min(line.points[0].y, line.points[1].y)
        if self.y >= minY - bounds and self.y <= maxY + bounds:
            maxX = max(line.points[0].x, line.points[1].x)
            minX = min(line.points[0].x, line.points[1].x)
            return self.x >= minX - bounds and self.x <= maxX + bounds
        return False

    def betweenLinePoints(self, line: 'Line') -> bool:
        """Check if point lies between line endpoints"""
        return self.onLine(line) and self.withinLineSquare(line)


class Line:
    """Class representing a line with intersection points"""
    def __init__(self, line: np.ndarray):
        x1, y1, x2, y2 = line[0]
        if y1 > y2:
            x2, y2, x1, y1 = line[0]
        self.points = [Point(x1, y1), Point(x2, y2)]
        self.angle = (self.points[1] - self.points[0]).angle()
        self.length = (self.points[1] - self.points[0]).mag()
        self.name = LineType.EMPTY
        self.intersections = []

    def __lt__(self, other: 'Line') -> bool:
        return self.angle < other.angle

    def __neg__(self) -> tuple:
        return self.points[0].x, self.points[0].y

    def __pos__(self) -> tuple:
        return self.points[1].x, self.points[1].y

    def __str__(self) -> str:
        return f"Line(name={self.name}, from={self.points[0]}, to={self.points[1]}, angle={self.angle})"

    def decompose(self) -> tuple:
        """Decompose line into slope and y-intercept (mx + b)"""
        p0, p1 = self.points
        if p0.x == p1.x:  # Handle vertical lines
            return float('inf'), p0.x
        a = (p0.y - p1.y) / (p0.x - p1.x)
        b = (p0.x * p1.y - p1.x * p0.y) / (p0.x - p1.x)
        return a, b

    def __mul__(self, other: 'Line') -> Point:
        """Calculate intersection point with another line"""
        l0a, l0b = self.decompose()
        l1a, l1b = other.decompose()

        # Handle parallel lines
        if l0a == l1a:
            return None

        # Handle vertical lines
        if l0a == float('inf'):
            x = l0b
            y = l1a * x + l1b
        elif l1a == float('inf'):
            x = l1b
            y = l0a * x + l0b
        else:
            # Find intersection using matrix solution
            A = np.array([[-l0a, 1], [-l1a, 1]])
            b = np.array([[l0b], [l1b]])
            x, y = np.round(np.squeeze(np.linalg.pinv(A) @ b), 4)

        return Point(x, y)

    def extendsLine(self, other: 'Line') -> bool:
        """Check if other line extends this line and update endpoints if true"""
        point0 = other.points[0]
        point1 = other.points[1]
        delOther = False
        if point0.onLine(self) and point1.onLine(self):
            if point0.y < self.points[0].y:
                self.points[0] = point0
            if point1.y > self.points[1].y:
                self.points[1] = point1
            delOther = True
        return delOther


class Helper:
    """Main helper class for football field analysis"""
    def __init__(self):
        self.last_valid_mask = None

    def detect_players(self, img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """Detect players in the image using color and variance masks"""
        # Color-based detection
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        # Variance-based detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, variance_mask = cv2.threshold(
            cv2.GaussianBlur(cv2.Laplacian(gray, cv2.CV_8U), (5, 5), 0),
            10, 255, cv2.THRESH_BINARY
        )

        # Combine masks
        player_mask = cv2.bitwise_or(blue_mask, red_mask)
        player_mask = cv2.bitwise_or(player_mask, black_mask)
        player_mask = cv2.bitwise_or(player_mask, variance_mask)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        player_mask = cv2.dilate(player_mask, kernel, iterations=2)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)

        return player_mask

    def create_grass_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create mask for grass field"""
        # Multiple grass color ranges for different lighting conditions
        grass_masks = [
            cv2.inRange(hsv, np.array([35, 50, 30]), np.array([85, 255, 200])),
            cv2.inRange(hsv, np.array([35, 30, 150]), np.array([85, 90, 255])),
            cv2.inRange(hsv, np.array([35, 50, 20]), np.array([85, 255, 150]))
        ]

        # Combine grass masks
        combined_mask = grass_masks[0]
        for mask in grass_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Filter out LED boards
        led_mask = cv2.inRange(hsv, np.array([0, 200, 200]), np.array([180, 255, 255]))
        kernel_horizontal = np.ones((1, 15), np.uint8)
        led_lines = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_horizontal)
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(led_lines))

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    def create_field_mask(self, img: np.ndarray) -> np.ndarray:
        """Create complete field mask including grass and removing players"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        field_mask = self.create_grass_mask(hsv)

        # Clean up mask
        kernel_medium = np.ones((7, 7), np.uint8)
        kernel_large = np.ones((21, 21), np.uint8)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_medium)

        # Find the largest contour (main field area)
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            field_mask = np.zeros_like(field_mask)
            cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)

        # Remove players from mask
        player_mask = self.detect_players(img, hsv)
        field_mask[player_mask > 0] = 0
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)

        # Use last valid mask if current is empty
        if np.sum(field_mask) > 0:
            self.last_valid_mask = field_mask.copy()
        elif self.last_valid_mask is not None:
            field_mask = self.last_valid_mask.copy()

        return field_mask

    def detect_lines(self, frame: np.ndarray, field_mask: np.ndarray) -> list:
        """Detect lines in the frame using field mask"""
        if field_mask is None:
            return None

        # Find ROI using field mask
        nz = cv2.findNonZero(field_mask)
        if nz is None:
            return None

        x, y, w, h = cv2.boundingRect(nz)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y + h, x:x + w]
        roi_mask = field_mask[y:y + h, x:x + w]
        roi_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)

        # Threshold based on mean value
        mean_val = cv2.mean(roi_masked, roi_mask)[0]
        _, roi_thresh = cv2.threshold(
            roi_masked,
            mean_val + 30,
            255,
            cv2.THRESH_BINARY
        )

        # Edge detection
        roi_thresh = cv2.GaussianBlur(roi_thresh, (5, 5), 0)
        edges = cv2.Canny(roi_thresh, 30, 150)

        # Line detection with HoughLinesP
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

        # Try with more lenient parameters if not enough lines found
        if lines is None or len(lines) < 2:
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=30,
                maxLineGap=40
            )

        if lines is not None:
            # Adjust coordinates back to original image space
            adjusted_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                adjusted_lines.append(np.array([[x1 + x, y1 + y, x2 + x, y2 + y]]))

            # Convert to Line objects
            return [Line(line) for line in adjusted_lines]

        return None

    def lineGrouper(self, lines: list) -> list:
        """Group lines by similar angles with 10-degree tolerance"""
        lines.sort()  # Sort by angle
        groups = []
        current_group = []
        group_base_angle = lines[0].angle

        for line in lines:
            if abs(line.angle - group_base_angle) < 10:
                current_group.append(line)
            else:
                group_base_angle = line.angle
                groups.append(current_group.copy())
                current_group = [line]

        if current_group:
            groups.append(current_group.copy())

        return groups

    def lineJoiner(self, grouped_lines: list) -> list:
        """Join lines within groups that are extensions of each other"""
        for group in grouped_lines:
            i = 0
            while i < len(group):
                j = i + 1
                while j < len(group):
                    if group[i].extendsLine(group[j]):
                        group.pop(j)
                    else:
                        j += 1
                i += 1
        return grouped_lines

    def lineFilter(self, lines: list, max_length: float, min_length: float,
                   min_angle: float, max_angle: float) -> list:
        """Filter lines by length and angle constraints"""
        return [
            line for line in lines
            if min_length < line.length < max_length
               and min_angle < line.angle < max_angle
        ]

    def createPointDict(self, lines: list) -> list:
        """Create dictionary of intersection points between vertical and horizontal lines"""
        points = []

        # Find intersections between vertical and horizontal lines
        for vertical in lines[0]:
            for horizontal in lines[1]:
                intersect = vertical * horizontal
                if intersect and intersect.betweenLinePoints(vertical) and intersect.betweenLinePoints(horizontal):
                    vertical.intersections.append(intersect)
                    horizontal.intersections.append(intersect)
                    intersect.interLines.append(vertical)
                    intersect.interLines.append(horizontal)
                    points.append(intersect)

        # Sort intersections by distance from line start
        for line in lines[0] + lines[1]:
            line.intersections.sort(key=lambda x: (line.points[0] - x).mag())

        return points

    def classifyLines(self, lines: list) -> None:
        """Classify lines and points based on field geometry"""
        # Find goal line and determine orientation
        goalGroup = -1
        horiGroup = -1
        goalLine = None

        # Look for goal line (line with 3+ intersections)
        for line in lines[0]:
            if len(line.intersections) >= 3:
                line.name = LineType.GOALLINE
                goalGroup = 0
                horiGroup = 1
                goalLine = line

        for line in lines[1]:
            if len(line.intersections) >= 3:
                line.name = LineType.GOALLINE
                goalGroup = 1
                horiGroup = 0
                goalLine = line

        if goalLine is None:
            return

        # Identify corner points and sidelines
        lCorner = None
        rCorner = None
        for line in lines[horiGroup]:
            if len(line.intersections) == 1:
                if line.angle - goalLine.angle < 180:
                    line.name = LineType.LEFTSIDELINE
                    line.intersections[0].name = PointType.LCORNER
                    lCorner = line.intersections[0]
                else:
                    line.name = LineType.RIGHTSIDELINE
                    line.intersections[0].name = PointType.RCORNER
                    rCorner = line.intersections[0]

        # Define point naming sequences
        LtoR = [PointType.LGOALLINEX16M, PointType.LGOALLINEX5M,
                PointType.RGOALLINEX5M, PointType.RGOALLINEX16M]
        RtoL = [PointType.RGOALLINEX16M, PointType.RGOALLINEX5M,
                PointType.LGOALLINEX5M, PointType.LGOALLINEX16M]

        # Process intersections based on which corner was identified first
        if goalLine.intersections[0].name == PointType.LCORNER:
            self._process_intersections_left_to_right(goalLine, LtoR)
        elif goalLine.intersections[0].name == PointType.RCORNER:
            self._process_intersections_right_to_left(goalLine, RtoL)

    def _process_intersections_left_to_right(self, goalLine: Line, point_sequence: list) -> None:
        """Process intersections from left to right"""
        for index, inter in enumerate(goalLine.intersections[1:]):
            inter.name = point_sequence[index]
            if inter.name == PointType.LGOALLINEX16M:
                self._process_16m_line_left(inter)
            elif inter.name == PointType.LGOALLINEX5M:
                self._process_5m_line_left(inter)

    def _process_intersections_right_to_left(self, goalLine: Line, point_sequence: list) -> None:
        """Process intersections from right to left"""
        for index, inter in enumerate(goalLine.intersections[1:]):
            inter.name = point_sequence[index]
            if inter.name == PointType.RGOALLINEX16M:
                self._process_16m_line_right(inter)
            elif inter.name == PointType.RGOALLINEX5M:
                self._process_5m_line_right(inter)

    def _process_16m_line_left(self, intersection: Point) -> None:
        """Process 16.5m line starting from left intersection"""
        for line in intersection.interLines:
            if line.name == LineType.EMPTY:
                line.name = LineType.LINE16L
                self._classify_line_points(line, PointType.LM16, LineType.LINE16,
                                           PointType.RM16, LineType.LINE16R)

    def _process_16m_line_right(self, intersection: Point) -> None:
        """Process 16.5m line starting from right intersection"""
        for line in intersection.interLines:
            if line.name == LineType.EMPTY:
                line.name = LineType.LINE16R
                self._classify_line_points(line, PointType.RM16, LineType.LINE16,
                                           PointType.LM16, LineType.LINE16L)

    def _process_5m_line_left(self, intersection: Point) -> None:
        """Process 5.5m line starting from left intersection"""
        for line in intersection.interLines:
            if line.name == LineType.EMPTY:
                line.name = LineType.LINE5L
                self._classify_line_points(line, PointType.LM5, LineType.LINE5,
                                           PointType.RM5, LineType.LINE5R)

    def _process_5m_line_right(self, intersection: Point) -> None:
        """Process 5.5m line starting from right intersection"""
        for line in intersection.interLines:
            if line.name == LineType.EMPTY:
                line.name = LineType.LINE5R
                self._classify_line_points(line, PointType.RM5, LineType.LINE5,
                                           PointType.LM5, LineType.LINE5L)

    def _classify_line_points(self, line: Line, first_point: PointType,
                              middle_line: LineType, second_point: PointType,
                              last_line: LineType) -> None:
        """Helper method to classify points and lines in sequence"""
        for intersection in line.intersections:
            if intersection.name == PointType.EMPTY:
                intersection.name = first_point
                for connected_line in intersection.interLines:
                    if connected_line.name == LineType.EMPTY:
                        connected_line.name = middle_line
                        for next_intersection in connected_line.intersections:
                            if next_intersection.name == PointType.EMPTY:
                                next_intersection.name = second_point
                                for final_line in next_intersection.interLines:
                                    if final_line.name == LineType.EMPTY:
                                        final_line.name = last_line
