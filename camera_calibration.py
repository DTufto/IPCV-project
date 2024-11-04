import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from helper import PointType, LineType, Point, Line, Helper
import colorsys

def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

def draw_field_lines(frame, lines_filtered, intersections):
    """Draw field lines and intersections"""
    output_frame = frame.copy()

    # Draw the lines
    for index, group in enumerate(lines_filtered):
        for line in group:
            color = hsv2rgb(index / 4, 0.5, 1)
            if line.name == LineType.GOALLINE:
                color = (255, 0, 0)  # Blue for goal line
            elif line.name in [LineType.LINE5L, LineType.LINE5, LineType.LINE5R]:
                color = (0, 255, 0)  # Green for 5m line
            elif line.name in [LineType.LINE16L, LineType.LINE16, LineType.LINE16R]:
                color = (255, 165, 0)  # Orange for 16m line
            cv2.line(output_frame,
                    (line.points[0].x, line.points[0].y),
                    (line.points[1].x, line.points[1].y),
                    color, 3)

    # Draw intersection points
    if intersections:
        for point in intersections:
            color = (0, 0, 255)  # Default red for general intersections

            # Color coding specific points
            if point.name in [PointType.LCORNER, PointType.RCORNER]:
                color = (255, 255, 0)  # Yellow for corners
            elif point.name in [PointType.LM5, PointType.RM5]:
                color = (0, 255, 0)  # Green for 5m points
            elif point.name in [PointType.LM16, PointType.RM16]:
                color = (255, 165, 0)  # Orange for 16m points

            cv2.circle(output_frame, (point.x, point.y), radius=10, color=color, thickness=-1)

            # Add text labels for points
            if point.name != PointType.EMPTY:
                cv2.putText(output_frame, point.name.name,
                            (point.x + 15, point.y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_frame

class FootballFieldCalibrator:
    def __init__(self, frame_width: int, frame_height: int):
        """Initialize calibrator with frame dimensions"""
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Define all possible world coordinates (in meters)
        self.world_points_dict = {
            # Goal line points
            PointType.LCORNER: (0, 34),       # Left corner
            PointType.RCORNER: (0, 0),        # Right corner
            PointType.LGOALLINEX16M: (0, 16.5),  # Left 16.5m line intersection
            PointType.LGOALLINEX5M: (0, 5.5),    # Left 5.5m line intersection
            PointType.RGOALLINEX5M: (0, 28.5),   # Right 5.5m line intersection
            PointType.RGOALLINEX16M: (0, 17.5),  # Right 16.5m line intersection

            # 5.5m box points
            PointType.LM5: (5.5, 5.5),        # Left 5.5m box corner
            PointType.RM5: (5.5, 28.5),       # Right 5.5m box corner

            # 16.5m box points
            PointType.LM16: (16.5, 16.5),     # Left 16.5m box corner
            PointType.RM16: (16.5, 17.5)      # Right 16.5m box corner
        }

        # Minimum number of points required for homography
        self.min_points_required = 5

        # For compatibility with existing code
        self.required_points = set()  # We'll populate this with any points we find

    def has_all_required_points(self, named_points: Dict[PointType, Tuple[int, int]]) -> bool:
        """Check if we have enough points for homography calculation"""
        return len(named_points) >= self.min_points_required

    def extract_named_points(self, intersections: List[Point]) -> Dict[PointType, Tuple[int, int]]:
        """Extract all valid points from intersection list"""
        named_points = {}
        for point in intersections:
            if point.name != PointType.EMPTY and point.name in self.world_points_dict:
                named_points[point.name] = (point.x, point.y)
                self.required_points.add(point.name)  # Add to required points set
        return named_points

    def calculate_homography(self, named_points: Dict[PointType, Tuple[int, int]]) -> Optional[np.ndarray]:
        """Calculate homography matrix from named points"""
        if not self.has_all_required_points(named_points):
            return None

        # Take the first 4 points if we have more
        world_points = []
        image_points = []

        for name, image_coord in list(named_points.items())[:4]:
            world_points.append(self.world_points_dict[name])
            image_points.append(image_coord)

        # Convert to numpy arrays
        world_points = np.array(world_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        try:
            H, status = cv2.findHomography(world_points, image_points, cv2.RANSAC, 5.0)
            return H if H is not None else None
        except cv2.error as e:
            print(f"Error calculating homography: {e}")
            return None


def process_football_frame_with_homography(frame, helper, calibrator):
    """Process frame and calculate homography if enough named points are found"""
    # Create field mask
    mask = helper.create_field_mask(frame.copy())
    if mask is None:
        return frame, None, [], []

    # Detect lines
    lines = helper.detect_lines(frame, mask)
    if lines is None or len(lines) < 2:
        return frame, None, [], []

    # Group and filter lines
    lines_grouped = helper.lineGrouper(lines)
    if len(lines_grouped) < 2:
        return frame, None, [], []

    lines_filtered = helper.lineJoiner(lines_grouped.copy())
    if len(lines_filtered) < 2:
        return frame, None, [], []

    try:
        # Get intersections and classify lines
        intersections = helper.createPointDict(lines_filtered)
        helper.classifyLines(lines_filtered)

        # Extract named points
        named_points = calibrator.extract_named_points(intersections)

        # Draw field lines and intersections
        debug_frame = draw_field_lines(frame, lines_filtered, intersections)

        # Add detection status
        found_points = len(named_points)
        total_points = len(calibrator.required_points) if calibrator.required_points else found_points
        status_color = (0, 255, 0) if found_points >= calibrator.min_points_required else (0, 165, 255)

        cv2.putText(debug_frame,
                   f"Named points: {found_points}/{total_points}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Only calculate homography if we have all required points
        homography = None
        if calibrator.has_all_required_points(named_points):
            homography = calibrator.calculate_homography(named_points)

        return debug_frame, homography, lines_filtered, intersections

    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, None, [], []

    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, None


def place_banner_with_homography(frame: np.ndarray, homography: np.ndarray,
                               banner: np.ndarray, blend: bool = True) -> np.ndarray:
    """Place banner on frame using pre-calculated homography"""
    h, w = frame.shape[:2]
    h_banner, w_banner = banner.shape[:2]

    # Define banner corners in world coordinates (meters)
    banner_world_points = np.array([
        [0, 16.5],  # Bottom left (at LGOALLINEX16M)
        [0, 5.5],   # Bottom right (at LGOALLINEX5M)
        [2, 5.5],   # Top right (2 meters up from bottom right)
        [2, 16.5]   # Top left (2 meters up from bottom left)
    ], dtype=np.float32)

    # Define banner corners in the banner image
    banner_points = np.array([
        [0, h_banner],        # Bottom left
        [w_banner, h_banner], # Bottom right
        [w_banner, 0],        # Top right
        [0, 0]               # Top left
    ], dtype=np.float32)

    # Calculate homography from banner to world coordinates
    H_banner = cv2.getPerspectiveTransform(banner_points, banner_world_points)

    # Combine homographies
    H_combined = homography @ H_banner

    # Warp banner into position
    warped_banner = cv2.warpPerspective(banner, H_combined, (w, h))

    if blend:
        # Create mask for blending
        mask = np.ones((banner.shape[0], banner.shape[1]), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, H_combined, (w, h))
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) / 255.0

        # Blend images
        result = frame * (1 - warped_mask) + warped_banner * warped_mask
        return result.astype(np.uint8)
    else:
        return warped_banner


def process_video(input_path: str, output_path: str, banner_path: str):
    """Two-pass video processing with line visualization and banner placement"""
    print("Pass 1: Finding stable homography...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize components
    helper = Helper()
    calibrator = FootballFieldCalibrator(frame_width, frame_height)

    # Load banner
    banner = cv2.imread(banner_path)
    if banner is None:
        print(f"Error loading banner from {banner_path}")
        return

    frame_count = 0
    stable_homography = None
    best_frame = None
    best_frame_lines = None
    best_frame_intersections = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            debug_frame, current_homography, lines_filtered, intersections = process_football_frame_with_homography(
                frame, helper, calibrator)

            if current_homography is not None:
                print(f"Found stable homography at frame {frame_count}/{total_frames}")
                stable_homography = current_homography
                best_frame = debug_frame
                best_frame_lines = lines_filtered
                best_frame_intersections = intersections
                break

            if frame_count % 30 == 0:
                print(f"Searching for homography: {frame_count}/{total_frames} frames processed")

    finally:
        cap.release()

    if stable_homography is None:
        print("Could not find a frame with all required points")
        return

    # Reset video capture for second pass
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Second pass - apply banner and draw lines for all frames
    print("\nPass 2: Processing full video with lines and banner...")

    frame_count = 0
    frames_with_lines = 0
    frames_with_banner = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame for line detection
            debug_frame, current_homography, lines_filtered, intersections = process_football_frame_with_homography(
                frame, helper, calibrator)

            # Start with debug frame that has lines drawn
            result_frame = debug_frame

            # Count frames where we detected lines
            if lines_filtered and intersections:
                frames_with_lines += 1

            # Apply banner using stable homography
            if stable_homography is not None:
                result_frame = place_banner_with_homography(result_frame, stable_homography, banner)
                frames_with_banner += 1

            # Add frame counter
            cv2.putText(result_frame,
                        f"Frame: {frame_count}/{total_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(result_frame)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(
        input_path="media/football_clips/clip4.mp4",
        output_path="output_with_lines_and_banner.mp4",
        banner_path="media/windows7_whopper.jpg"
    )