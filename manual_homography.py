import cv2
import numpy as np
from camera_calibration import draw_field_lines
from helper import Helper


def get_ground_homography(clip_num: int) -> np.ndarray:
    """Get homography matrix for ground plane transformation for specific clip."""
    # Define ground points only (z=0) for each clip
    clip_data = {
        1: {
            'world_points': np.array([
                [50, 50],  # Point 1
                [189, 50],  # Point 2
                [298, 50],  # Point 3
                [353, 50],  # Point 4
                [426, 50],  # Point 6
                [298, 105],  # Point 8
                [481, 105]  # Point 9
            ], dtype=np.float32),
            'image_points': np.array([
                [364, 341],  # Point 1
                [792, 441],  # Point 2
                [1212, 537],  # Point 3
                [1459, 595],  # Point 4
                [1851, 684],  # Point 6
                [937, 581],  # Point 8
                [1877, 828]  # Point 9
            ], dtype=np.float32)
        },
        2: {
            'world_points': np.array([
                [50, 50],  # Point 1
                [189, 50],  # Point 2
                [298, 50],  # Point 3
                [353, 50],  # Point 4
                [426, 50],  # Point 6
                [298, 105],  # Point 8
                [481, 105]  # Point 9
            ], dtype=np.float32),
            'image_points': np.array([
                [343, 330],  # Point 1
                [740, 449],  # Point 2
                [1123, 563],  # Point 3
                [1343, 629],  # Point 4
                [1677, 733],  # Point 6
                [779, 601],  # Point 8
                [1563, 874]  # Point 9
            ], dtype=np.float32)
        },
        3: {
            'world_points': np.array([
                [50, 50],  # Point 1
                [189, 50],  # Point 2
                [298, 50],  # Point 3
                [353, 50],  # Point 4
                [426, 50],  # Point 6
                [298, 105],  # Point 8
                [481, 105],  # Point 9
                [189, 215]  # Point 10
            ], dtype=np.float32),
            'image_points': np.array([
                [1220, 163],  # Point 1
                [1340, 344],  # Point 2
                [1458, 523],  # Point 3
                [1527, 630],  # Point 4
                [1641, 800],  # Point 6
                [1095, 532],  # Point 8
                [1279, 957],  # Point 9
                [330, 362]  # Point 10
            ], dtype=np.float32)
        }
    }

    data = clip_data[clip_num]

    # Calculate homography from world to image coordinates
    H, _ = cv2.findHomography(data['world_points'], data['image_points'])
    return H


def place_banner_with_homography(frame: np.ndarray, banner: np.ndarray,
                                 homography: np.ndarray, blend: bool = True) -> np.ndarray:
    """Place banner using homography transformation."""
    h, w = frame.shape[:2]
    h_banner, w_banner = banner.shape[:2]

    # Define banner corners in world coordinates (meters)
    banner_world_points = np.array([
        [50, 70],  # Bottom left
        [189, 70],  # Bottom right
        [189, 90],  # Top right
        [50, 90]  # Top left
    ], dtype=np.float32)

    # Define banner corners in the banner image
    banner_points = np.array([
        [0, h_banner],  # Bottom left
        [w_banner, h_banner],  # Bottom right
        [w_banner, 0],  # Top right
        [0, 0]  # Top left
    ], dtype=np.float32)

    # Calculate homography from banner to world coordinates
    H_banner = cv2.getPerspectiveTransform(banner_points, banner_world_points)

    # Combine homographies to go directly from banner to image
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


def process_video_with_homography(
        clip_num: int,
        input_path: str,
        output_path: str,
        banner_path: str,
        draw_lines: bool = True
):
    """Process video using homography-based banner placement."""
    # Initialize video capture
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
    homography = get_ground_homography(clip_num)

    # Load banner
    banner = cv2.imread(banner_path)
    if banner is None:
        raise ValueError(f"Could not load banner image: {banner_path}")

    # Initialize video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0
    frames_with_lines = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result_frame = frame.copy()

            if draw_lines:
                # Detect and draw field lines
                mask = helper.create_field_mask(frame.copy())
                if mask is not None:
                    lines = helper.detect_lines(frame, mask)
                    if lines is not None and len(lines) >= 2:
                        lines_grouped = helper.lineGrouper(lines)
                        if len(lines_grouped) >= 2:
                            lines_filtered = helper.lineJoiner(lines_grouped.copy())
                            if len(lines_filtered) >= 2:
                                intersections = helper.createPointDict(lines_filtered)
                                helper.classifyLines(lines_filtered)
                                result_frame = draw_field_lines(
                                    result_frame,
                                    lines_filtered,
                                    intersections
                                )
                                frames_with_lines += 1

            # Place banner using homography
            result_frame = place_banner_with_homography(
                result_frame,
                banner,
                homography,
                blend=True
            )

            out.write(result_frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Process each clip
    for clip_num in [1, 2, 3]:
        process_video_with_homography(
            clip_num=clip_num,
            input_path=f"media/football_clips/clip{clip_num}.mp4",
            output_path=f"output_clip{clip_num}_homography.mp4",
            banner_path="media/windows7_whopper.jpg",
            draw_lines=True
        )