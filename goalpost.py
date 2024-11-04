import cv2
import numpy as np
from helper import Helper, Line
from typing import Tuple, Dict, List, Optional


class GoalpostDetector:
    def __init__(self):
        # Relative size parameters (as fractions of frame dimensions)
        self.MIN_HEIGHT_RATIO = 0.15  # Minimum height as fraction of frame height
        self.MAX_WIDTH_RATIO = 0.015  # Maximum width as fraction of frame width

    def detect_white_objects(self, frame: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        """
        Detect white objects using adaptive thresholds based on field characteristics
        """
        # Convert to different color spaces for robust detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate adaptive thresholds based on field brightness
        field_mean = cv2.mean(gray, mask=field_mask)[0]
        field_std = cv2.meanStdDev(gray, mask=field_mask)[1][0][0]

        # White objects should be significantly brighter than the field
        brightness_threshold = min(255, int(field_mean + 2 * field_std))

        # Create white mask using multiple techniques
        # 1. HSV-based detection with explicit uint8 type
        white_hsv = cv2.inRange(
            hsv,
            np.array([0, 0, brightness_threshold], dtype=np.uint8),
            np.array([180, 30, 255], dtype=np.uint8)
        )

        # 2. Adaptive thresholding on grayscale
        white_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, -5)

        # Combine masks
        white_mask = cv2.bitwise_and(white_hsv, white_adaptive)

        # Clean up noise while preserving vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, vertical_kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, horizontal_kernel)

        return white_mask


    def extract_goalpost_points(self, contour: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Extract top and bottom points from a goalpost contour
        """
        # Get the highest and lowest points of the contour
        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
        return top_point, bottom_point

    def find_goalpost_candidates(self, contours: List[np.ndarray], frame_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find potential goalpost contours
        """
        frame_height, frame_width = frame_shape[:2]
        min_height = int(frame_height * 0.1)  # Increased to 10% of frame height
        max_width = int(frame_width * 0.02)  # Reduced to 2% of frame width

        candidates = []
        debug_rejects = {'angle': 0, 'height': 0, 'width': 0, 'ratio': 0}

        for idx, contour in enumerate(contours):
            if len(contour) < 3:
                continue

            # basic measurements
            x, y, w, h = cv2.boundingRect(contour)

            # rotated rectangle for angle analysis
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            height = max(rect[1])
            angle = abs(rect[2] if rect[2] < 45 else 90 - rect[2])

            is_candidate = True
            reason = []

            # Height check
            if height < min_height:
                is_candidate = False
                reason.append(f"height {height:.1f} < {min_height}")
                debug_rejects['height'] += 1

            # Width check
            if width > max_width:
                is_candidate = False
                reason.append(f"width {width:.1f} > {max_width}")
                debug_rejects['width'] += 1

            # angle check
            if angle > 20:  # Reduced from 30 to 20 degrees
                is_candidate = False
                reason.append(f"angle {angle:.1f} > 20")
                debug_rejects['angle'] += 1

            # aspect ratio check
            if width > 0 and height / width < 3:  # Increased from 1.5 to 3
                is_candidate = False
                reason.append(f"aspect {height / width:.1f} < 3")
                debug_rejects['ratio'] += 1

            # Print debug info for larger contours
            if height > frame_height * 0.05:
                print(f"\nContour {idx}:")
                print(f"  Height: {height:.1f} px ({height / frame_height * 100:.1f}% of frame)")
                print(f"  Width: {width:.1f} px ({width / frame_width * 100:.1f}% of frame)")
                print(f"  Angle: {angle:.1f}°")
                print(f"  Aspect Ratio: {height / width if width > 0 else 0:.1f}")
                if not is_candidate:
                    print(f"  Rejected due to: {', '.join(reason)}")
                else:
                    print("  ACCEPTED")

            if is_candidate:
                candidates.append(contour)

        print("\nDetection Summary:")
        print(f"Total contours analyzed: {len(contours)}")
        print(f"Candidates found: {len(candidates)}")
        print("\nRejection reasons:")
        for reason, count in debug_rejects.items():
            print(f"  {reason}: {count}")

        return candidates

    def detect_goalposts(self, frame: np.ndarray, field_mask: np.ndarray, helper: Helper) -> Tuple[List, Dict]:
        """
        Detect goalposts and draw field lines
        """
        debug_dict = {}

        # Create search area around the field
        search_area = cv2.dilate(field_mask, np.ones((101, 101), np.uint8), iterations=2)

        # Detect white objects
        white_mask = self.detect_white_objects(frame, field_mask)
        white_mask = cv2.bitwise_and(white_mask, search_area)

        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create visualization with all contours
        contour_vis = frame.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 255), 2)

        # Find goalpost candidates
        candidates = self.find_goalpost_candidates(contours, frame.shape)

        # Add candidate visualization
        candidate_vis = frame.copy()
        for idx, candidate in enumerate(candidates):
            x, y, w, h = cv2.boundingRect(candidate)
            cv2.drawContours(candidate_vis, [candidate], -1, (0, 255, 0), 2)
            cv2.putText(candidate_vis, f"#{idx}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process detected goalposts
        goalpost_rects = []
        debug_frame = frame.copy()

        # Detect lines using your existing line detection
        lines = helper.detect_lines_on_field(frame, field_mask)
        if lines is not None:
            # Convert to Line objects and group them
            lines = list(map(Line, lines))
            grouped_lines = helper.lineGrouper(lines)

            # Draw lines
            for group in grouped_lines:
                color = (255, 0, 255)  # Magenta for field lines
                for line in group:
                    cv2.line(debug_frame, -line, +line, color, 2)

            # Find and draw intersections
            intersections = helper.find_field_intersections(grouped_lines, field_mask)
            for point in intersections:
                cv2.circle(debug_frame, point, 3, (0, 255, 255), -1)  # Yellow for intersections

        # Draw goalposts on top of lines
        for idx, post in enumerate(candidates):
            x, y, w, h = cv2.boundingRect(post)
            goalpost_rects.append([x, y, w, h])

            # Get top and bottom points
            top_point, bottom_point = self.extract_goalpost_points(post)

            # Draw detection visualization
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
            cv2.circle(debug_frame, top_point, 5, (255, 0, 0), -1)  # Blue top point
            cv2.circle(debug_frame, bottom_point, 5, (0, 255, 0), -1)  # Green bottom point
            cv2.putText(debug_frame, f"Post {idx + 1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add debug visualizations
        debug_dict['white_mask'] = self.create_debug_image(
            cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR),
            "White Objects")
        debug_dict['contours'] = self.create_debug_image(
            contour_vis,
            "All Contours")
        debug_dict['candidates'] = self.create_debug_image(
            candidate_vis,
            f"Candidates ({len(candidates)})")
        debug_dict['detection_result'] = self.create_debug_image(
            debug_frame,
            f"Goalpost & Line Detection")

        return goalpost_rects, debug_dict

    def create_debug_image(self, frame: np.ndarray, name: str, text: str = "") -> np.ndarray:
        """Create a debug image"""
        debug_img = frame.copy()
        cv2.rectangle(debug_img, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(debug_img, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if text:
            cv2.putText(debug_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return debug_img


def create_grid_visualization(frames_dict: dict) -> np.ndarray:
    """Create a 2x2 grid visualization of detection steps"""
    # Get dimensions of first frame
    frame = next(iter(frames_dict.values()))
    if isinstance(frame, tuple):  # Handle case where frame is a tuple with title
        frame = frame[0]
    h, w = frame.shape[:2]

    # Calculate target size (scale down to 720p equivalent)
    target_height = 720
    scale = target_height / (h * 2)  # Scale to fit two frames vertically in 720p
    target_width = int(w * scale)
    target_height = int(h * scale)

    # Create blank canvas for 2x2 grid
    grid = np.zeros((target_height * 2, target_width * 2, 3), dtype=np.uint8)

    # Position for each frame in the grid
    positions = {
        'original': (0, 0),
        'field_mask': (0, 1),
        'white_mask': (1, 0),
        'detection_result': (1, 1)
    }

    # Place each frame in its position
    for name, frame in frames_dict.items():
        if frame is None:
            continue

        # Handle case where frame is a tuple with title
        if isinstance(frame, tuple):
            frame = frame[0]

        # Convert single channel images to BGR
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Get grid position
        row, col = positions[name]
        grid[row * target_height:(row + 1) * target_height,
        col * target_width:(col + 1) * target_width] = resized_frame

        # Add label with smaller font size
        cv2.putText(grid, name.replace('_', ' ').title(),
                    (col * target_width + 10, row * target_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return grid


def main():
    # Initialize helper and detector
    helper = Helper()
    detector = GoalpostDetector()

    # Open video file
    video_path = 'media/football_clips/clip3.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties for input
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate output dimensions (scaled to 720p equivalent)
    output_height = 720
    scale = output_height / (frame_height * 2)
    output_width = int(frame_width * 2 * scale)
    output_height = int(frame_height * 2 * scale)

    # Create video writer for output with scaled dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('goalpost_detection.mp4', fourcc, fps,
                          (output_width, output_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create field mask
            field_mask = helper.create_field_mask(frame)

            # Detect goalposts
            goalposts, debug_info = detector.detect_goalposts(frame, field_mask, helper)

            # Create visualization grid
            vis_frames = {
                'original': frame,
                'field_mask': field_mask,
                'white_mask': debug_info['white_mask'],
                'detection_result': debug_info['detection_result']
            }

            grid = create_grid_visualization(vis_frames)

            # Add detection information
            info_text = f"Detected {len(goalposts)} goalposts"
            cv2.putText(grid, info_text, (10, grid.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show and write frame
            cv2.imshow('Goalpost Detection', grid)
            out.write(grid)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Processing complete!")


if __name__ == "__main__":
    main()