import cv2
import numpy as np
from helper import Helper
from penalty_spot import PenaltySpotDetector
from pathlib import Path


def create_debug_window(name, images_dict, scale=0.3):
    """Create a debug window with multiple images in a grid"""
    # Determine grid size
    n_images = len(images_dict)
    cols = 4  # Fixed number of columns
    rows = (n_images + cols - 1) // cols

    # Get dimensions of first image
    first_image = next(iter(images_dict.values()))
    h, w = first_image.shape[:2] if len(first_image.shape) > 1 else (first_image.shape[0], first_image.shape[0])

    # Add padding between images
    padding = 10
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)

    # Create canvas with padding
    canvas_h = scaled_h * rows + padding * (rows + 1)
    canvas_w = scaled_w * cols + padding * (cols + 1)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images
    for idx, (title, img) in enumerate(images_dict.items()):
        row = idx // cols
        col = idx % cols

        # Convert to BGR if grayscale
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()

        # Resize
        img_resized = cv2.resize(img_bgr, (scaled_w, scaled_h))

        # Calculate position with padding
        y1 = row * scaled_h + padding * (row + 1)
        y2 = y1 + scaled_h
        x1 = col * scaled_w + padding * (col + 1)
        x2 = x1 + scaled_w

        # Place image
        canvas[y1:y2, x1:x2] = img_resized

        # Add title (smaller font)
        cv2.putText(canvas, title, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(name, canvas)


def debug_penalty_spot_detection(frame):
    """Debug penalty spot detection with visualizations"""
    # Initialize components
    helper = Helper()
    detector = PenaltySpotDetector()

    # Create field mask
    field_mask = helper.create_field_mask(frame)
    if field_mask is None:
        print("Error: Could not create field mask")
        return

    # Create white mask for spot detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 30, 255]))

    # Get penalty area mask
    penalty_area_mask = detector._create_penalty_area_mask(field_mask)

    # Apply masks sequentially
    spot_search_area = cv2.bitwise_and(white_mask, penalty_area_mask)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_spot_area = cv2.morphologyEx(spot_search_area, cv2.MORPH_OPEN, kernel)

    # Find circles
    circles = cv2.HoughCircles(
        cleaned_spot_area,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=10,
        minRadius=3,
        maxRadius=15
    )

    # Create visualization of detected circles
    circles_vis = frame.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            score = detector._evaluate_penalty_spot(x, y, r, cleaned_spot_area)

            # Only draw high-confidence detections
            if score > 0.5:
                cv2.circle(circles_vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(circles_vis, (x, y), 2, (0, 0, 255), 3)
                # Move score text above circle
                cv2.putText(circles_vis, f"{score:.2f}", (x - 20, y - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Create combined mask visualization
    combined_mask = cv2.addWeighted(
        cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR), 0.5,
        cv2.cvtColor(penalty_area_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Display all intermediate results
    debug_images = {
        'Frame': frame,
        'Field Mask': field_mask,
        'White Mask': white_mask,
        'Penalty Area': penalty_area_mask,
        'Combined Masks': combined_mask,
        'Search Area': spot_search_area,
        'Cleaned Area': cleaned_spot_area,
        'Detections': circles_vis
    }

    create_debug_window('Penalty Spot Detection Debug', debug_images)

    # Print compact debug info
    print(f"\nFrame Analysis:")
    print(f"- Field mask pixels: {np.count_nonzero(field_mask):,}")
    print(f"- Search area pixels: {np.count_nonzero(cleaned_spot_area):,}")
    if circles is not None:
        high_conf = sum(1 for circle in circles[0] if
                        detector._evaluate_penalty_spot(circle[0], circle[1], circle[2],
                                                        cleaned_spot_area) > 0.5)
        print(f"- Detected spots: {high_conf} (high confidence) / {len(circles[0])} (total)")


def process_video_clips(clips_folder: str):
    """Process all video clips in the specified folder"""
    clips_path = Path(clips_folder)
    video_files = sorted(list(clips_path.glob('*.mp4')))

    if not video_files:
        print(f"No MP4 files found in {clips_folder}")
        return

    print("\nControls:")
    print("- Space: Next frame")
    print("- n: Next video")
    print("- q: Quit")
    print("\nProcessing videos...")

    for video_file in video_files:
        print(f"\nProcessing {video_file}")
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"\nFrame {frame_count}")
            debug_penalty_spot_detection(frame)

            key = cv2.waitKey(0)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == 32:  # Space
                frame_count += 1
                continue

        cap.release()

    cv2.destroyAllWindows()
    print("\nProcessing complete!")


if __name__ == "__main__":
    process_video_clips('./media/football_clips/')