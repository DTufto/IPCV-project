import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
from scipy import stats
from helper import Helper, Line
from camera_calibration import CameraCalibrator
from football_liner import process_football_frame
from dataclasses import dataclass


@dataclass
class TrackedPoint:
    position: Tuple[int, int]
    frames_seen: int
    last_seen: int
    total_movement: float  # Track total movement to detect jitter


class PointTracker:
    def __init__(self, max_distance: int = 10, history_frames: int = 5,
                 min_seen_frames: int = 3, max_movement: float = 20.0):
        """
        Initialize point tracker
        max_distance: Maximum pixel distance to consider points the same
        history_frames: Number of frames to keep in history
        min_seen_frames: Minimum frames a point must be seen to be considered valid
        max_movement: Maximum total movement allowed for a point to be considered stable
        """
        self.max_distance = max_distance
        self.history_frames = history_frames
        self.min_seen_frames = min_seen_frames
        self.max_movement = max_movement

        self.tracked_points: Dict[int, TrackedPoint] = {}
        self.current_frame = 0
        self.next_id = 0

    def update(self, new_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Update tracked points with new detections
        Returns list of stable points for this frame
        """
        self.current_frame += 1
        matched_ids = set()

        # Match new points to existing tracked points
        for new_point in new_points:
            matched_id = self._match_point(new_point)
            if matched_id is not None:
                matched_ids.add(matched_id)
                point = self.tracked_points[matched_id]

                # Update movement
                movement = np.sqrt(
                    (point.position[0] - new_point[0]) ** 2 +
                    (point.position[1] - new_point[1]) ** 2
                )
                point.total_movement += movement

                # Update position and counters
                point.position = new_point
                point.frames_seen += 1
                point.last_seen = self.current_frame
            else:
                # Create new tracked point
                self.tracked_points[self.next_id] = TrackedPoint(
                    position=new_point,
                    frames_seen=1,
                    last_seen=self.current_frame,
                    total_movement=0.0
                )
                self.next_id += 1

        # Remove old points
        self._cleanup_old_points()

        # Return stable points
        return self.get_stable_points()

    def _match_point(self, new_point: Tuple[int, int]) -> Optional[int]:
        """Find closest existing point within threshold"""
        best_dist = float('inf')
        best_id = None

        for point_id, tracked in self.tracked_points.items():
            if tracked.last_seen == self.current_frame:
                continue  # Skip already matched points

            dist = np.sqrt(
                (tracked.position[0] - new_point[0]) ** 2 +
                (tracked.position[1] - new_point[1]) ** 2
            )

            if dist < self.max_distance and dist < best_dist:
                best_dist = dist
                best_id = point_id

        return best_id

    def _cleanup_old_points(self):
        """Remove points not seen recently"""
        to_remove = []
        for point_id, point in self.tracked_points.items():
            if self.current_frame - point.last_seen > self.history_frames:
                to_remove.append(point_id)

        for point_id in to_remove:
            del self.tracked_points[point_id]

    def get_stable_points(self) -> List[Tuple[int, int]]:
        """Return list of points that are considered stable"""
        stable_points = []

        for point in self.tracked_points.values():
            if (point.frames_seen >= self.min_seen_frames and
                    point.total_movement / point.frames_seen < self.max_movement):
                stable_points.append(point.position)

        return stable_points


class CameraStabilizer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.camera_matrices = []
        self.dist_coeffs = []

    def add_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """Add new camera parameters"""
        if camera_matrix is not None and dist_coeffs is not None:
            self.camera_matrices.append(camera_matrix)
            self.dist_coeffs.append(dist_coeffs)

    def _matrix_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Compute similarity score between two camera matrices
        Lower score means more similar
        """
        # Normalize matrices to account for scale differences
        norm1 = matrix1 / np.linalg.norm(matrix1)
        norm2 = matrix2 / np.linalg.norm(matrix2)

        # Compute Frobenius norm of difference
        return np.linalg.norm(norm1 - norm2)

    def _find_similar_matrices(self, matrices: List[np.ndarray], keep_percent: float = 0.5) -> List[int]:
        """
        Find indices of matrices that are most similar to each other
        keep_percent: percentage of matrices to keep (0.0 to 1.0)
        """
        n_matrices = len(matrices)
        if n_matrices < 2:
            return list(range(n_matrices))

        # Compute similarity scores for each pair
        similarities = np.zeros((n_matrices, n_matrices))
        for i in range(n_matrices):
            for j in range(i + 1, n_matrices):
                score = self._matrix_similarity(matrices[i], matrices[j])
                similarities[i, j] = score
                similarities[j, i] = score

        # For each matrix, compute average similarity to all others
        avg_similarities = np.mean(similarities, axis=1)

        # Keep the matrices with lowest average similarity scores
        keep_count = max(2, int(n_matrices * keep_percent))
        best_indices = np.argsort(avg_similarities)[:keep_count]

        return sorted(best_indices)

    def compute_stable_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute stable camera parameters from most similar 50%"""
        if len(self.camera_matrices) < 2:
            return None, None

        # Find most similar matrices
        best_indices = self._find_similar_matrices(self.camera_matrices)

        if not best_indices:
            return None, None

        # Average the selected matrices
        selected_cameras = [self.camera_matrices[i] for i in best_indices]
        selected_dists = [self.dist_coeffs[i] for i in best_indices]

        stable_camera = np.mean(selected_cameras, axis=0)
        stable_dist = np.mean(selected_dists, axis=0)

        print(f"Stabilized using {len(best_indices)} similar frames out of {len(self.camera_matrices)}")

        return stable_camera, stable_dist


class GroundBannerPlacer:
    def __init__(self, banner_path: str, banner_width_meters: float, banner_height_meters: float):
        """
        Initialize banner placer with hybrid positioning
        """
        self.banner = cv2.imread(banner_path)
        if self.banner is None:
            raise ValueError(f"Could not load banner image from {banner_path}")

        self.banner_width = banner_width_meters
        self.banner_height = banner_height_meters
        self.initial_position = None

    def _find_ground_plane_position(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                    rvec: np.ndarray, tvec: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """Find a suitable position on the ground plane for banner placement"""
        h_frame, w_frame = frame_shape[:2]

        # Create a grid of test points on the ground plane (Y=0)
        # Testing points from -30m to +30m on X and Z axes
        x = np.linspace(-30, 30, 30)
        z = np.linspace(-30, 30, 30)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)

        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        points = points.astype(np.float32)

        # Project points to image plane
        projected_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)

        # Find points that project to middle-lower part of frame
        target_y = h_frame * 0.7  # Aim for lower third of frame
        target_x = w_frame * 0.5  # Center horizontally

        distances = np.sqrt(
            (projected_points[:, 0] - target_x) ** 2 +
            (projected_points[:, 1] - target_y) ** 2
        )

        # Find the ground point that projects closest to our target position
        best_idx = np.argmin(distances)
        best_point = points[best_idx]

        return best_point

    def _get_banner_corners(self, center_point: np.ndarray) -> np.ndarray:
        """Calculate banner corners in world coordinates around a center point"""
        x, y, z = center_point
        half_width = self.banner_width / 2

        # Place banner corners parallel to X axis
        return np.float32([
            [x - half_width, y, z],  # Bottom left
            [x + half_width, y, z],  # Bottom right
            [x + half_width, y + self.banner_height, z],  # Top right
            [x - half_width, y + self.banner_height, z]  # Top left
        ])

    def place_banner(self, frame: np.ndarray, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                     helper: MergedHelper) -> np.ndarray:
        """
        Place banner using homography for ground plane and line detection for positioning
        """
        h_frame, w_frame = frame.shape[:2]
        result = frame.copy()

        try:
            # First get the goal line
            field_mask = helper.create_field_mask(frame)
            if field_mask is None:
                return frame

            lines = helper.detect_lines(frame, field_mask)
            if lines is None:
                return frame

            grouped_lines = helper.lineGrouper(lines)
            filtered_lines = helper.lineJoiner(grouped_lines)
            goal_line = self.find_goal_line(filtered_lines)

            if goal_line is None:
                return frame

            # Define banner corners in world coordinates (meters)
            banner_world_points = np.array([
                [0, 0, 0],  # Bottom left
                [self.banner_width, 0, 0],  # Bottom right
                [self.banner_width, 0, self.banner_height],  # Top right
                [0, 0, self.banner_height]  # Top left
            ], dtype=np.float32)

            # Project world points to image plane
            projected_points, _ = cv2.projectPoints(
                banner_world_points, rvec, tvec, camera_matrix, dist_coeffs)
            projected_points = projected_points.reshape(-1, 2)

            # If this is the first frame, calculate initial offset from goal line
            if self.initial_position is None:
                self.initial_position = self.calculate_offset_from_line(
                    projected_points, goal_line)
                print("Initial banner offset from goal line:", self.initial_position)

            # Adjust banner position to maintain consistent offset from goal line
            line_start = np.array([goal_line.points[0].x, goal_line.points[0].y])
            line_end = np.array([goal_line.points[1].x, goal_line.points[1].y])
            line_mid = (line_start + line_end) / 2

            # Apply stored offset to maintain relative position
            adjustment = line_mid + self.initial_position - np.mean(projected_points[:2], axis=0)
            projected_points += adjustment

            # Get banner image corners
            h_banner, w_banner = self.banner.shape[:2]
            banner_corners = np.float32([
                [0, h_banner],  # Bottom left
                [w_banner, h_banner],  # Bottom right
                [w_banner, 0],  # Top right
                [0, 0]  # Top left
            ])

            # Compute homography for the banner
            H, _ = cv2.findHomography(banner_corners, projected_points)
            if H is None:
                return frame

            # Warp banner
            warped_banner = cv2.warpPerspective(self.banner, H, (w_frame, h_frame))

            # Create mask for blending
            mask = np.ones((h_banner, w_banner), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H, (w_frame, h_frame))
            mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) / 255.0

            # Blend images
            result = frame * (1 - mask_3d) + warped_banner * mask_3d

            # Optional: Draw debug visualization
            if False:  # Set to True to see debug overlay
                # Draw goal line
                cv2.line(result,
                         (int(line_start[0]), int(line_start[1])),
                         (int(line_end[0]), int(line_end[1])),
                         (0, 255, 0), 2)
                # Draw banner corners
                for point in projected_points:
                    cv2.circle(result,
                               (int(point[0]), int(point[1])),
                               5, (0, 0, 255), -1)

        except Exception as e:
            print(f"Error placing banner: {e}")
            return frame

        return result.astype(np.uint8)


def process_football_frame_with_tracking(frame, helper, calibrator, point_tracker):
    """Process frame with point tracking"""
    field_mask = helper.create_field_mask(frame)
    if field_mask is None:
        return frame, None, []

    # Detect lines
    lines = helper.detect_lines_on_field(frame, field_mask)
    if lines is None:
        return frame, None, []

    lines = list(map(Line, lines))
    grouped_lines = helper.lineGrouper(lines)

    # Find intersections
    current_intersections = helper.find_field_intersections(grouped_lines, field_mask)

    # Update point tracker and get stable points
    stable_points = point_tracker.update(current_intersections)

    # Only attempt calibration if we have enough stable points
    camera_params = None
    if len(stable_points) >= 8:
        camera_params = calibrator.process_frame(
            frame, field_mask, stable_points, grouped_lines, helper)

    # Create debug visualization
    result_frame = frame.copy()

    # Draw all detected intersections in red
    for point in current_intersections:
        cv2.circle(result_frame, point, 3, (0, 0, 255), -1)

    # Draw stable points in green
    for point in stable_points:
        cv2.circle(result_frame, point, 5, (0, 255, 0), 2)

    # Add debug text
    cv2.putText(result_frame, f"Current points: {len(current_intersections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(result_frame, f"Stable points: {len(stable_points)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_frame, camera_params, stable_points


def process_video(input_path: str, output_path: str, banner_path: str,
                  banner_width_meters: float = 10.0, banner_height_meters: float = 2.0):
    """
    Process video with banner placement on all frames using stable camera intrinsics
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize components
    helper = Helper()
    calibrator = CameraCalibrator(width, height)
    stabilizer = CameraStabilizer()
    point_tracker = PointTracker()
    banner_placer = GroundBannerPlacer(banner_path, banner_width_meters, banner_height_meters)

    # First pass: collect camera parameters and poses
    print("Pass 1: Computing stable camera parameters...")

    frame_poses = {}  # Store poses for each frame
    valid_frames = 0
    frame_idx = 0
    last_valid_pose = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with point tracking
            _, camera_params, _ = process_football_frame_with_tracking(
                frame, helper, calibrator, point_tracker)

            if camera_params is not None:
                camera_matrix, dist_coeffs, pose = camera_params
                if camera_matrix is not None and dist_coeffs is not None and pose is not None:
                    stabilizer.add_camera_params(camera_matrix, dist_coeffs)
                    frame_poses[frame_idx] = pose
                    last_valid_pose = pose
                    valid_frames += 1

            # For frames without calibration, use the last valid pose
            elif last_valid_pose is not None:
                frame_poses[frame_idx] = last_valid_pose

            if frame_idx % 30 == 0:
                print(f"Pass 1: Processed {frame_idx}/{total_frames} frames ({valid_frames} valid)")

            frame_idx += 1

    finally:
        cap.release()

    # Get stable camera parameters
    stable_matrix, stable_dist = stabilizer.compute_stable_params()
    if stable_matrix is None or stable_dist is None:
        print("Failed to compute stable camera parameters")
        return

    # Second pass: apply banner to all frames
    print("\nPass 2: Placing banner on all frames...")

    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    banners_placed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result_frame = frame.copy()

            # Get pose for this frame (either actual or interpolated)
            pose = frame_poses.get(frame_idx, last_valid_pose)
            if pose is not None:
                rvec, tvec = pose
                try:
                    result_frame = banner_placer.place_banner(
                        frame, stable_matrix, stable_dist, rvec, tvec)
                    banners_placed += 1
                except Exception as e:
                    print(f"Error placing banner in frame {frame_idx}: {e}")

            out.write(result_frame)

            if frame_idx % 30 == 0:
                print(f"Pass 2: Processed {frame_idx}/{total_frames} frames ({banners_placed} banners placed)")

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Print summary
        print("\nProcessing complete!")
        print(f"Total frames processed: {frame_idx}")
        print(f"Frames with valid calibration: {valid_frames}")
        print(f"Frames with banner placed: {banners_placed}")
        print(f"Calibration success rate: {(valid_frames / frame_idx) * 100:.1f}%")
        print(f"Banner placement success rate: {(banners_placed / frame_idx) * 100:.1f}%")


if __name__ == "__main__":
    process_video(
        input_path="media/football_clips/clip2.mp4",
        output_path="output_with_banner.mp4",
        banner_path="media/windows7_whopper.jpg",
        banner_width_meters=10.0,  # 10 meters wide
        banner_height_meters=2.0  # 2 meters tall
    )