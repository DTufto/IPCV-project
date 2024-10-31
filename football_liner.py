import cv2
from helper import Helper, Line
from pathlib import Path


def process_football_frame(frame, helper):
    # Create field mask
    field_mask = helper.create_field_mask(frame)

    # Detect lines using improved white line detection
    lines = helper.detect_lines_on_field(frame, field_mask)

    # If no lines detected, return original frame
    if lines is None:
        return frame

    # Convert to Line objects and group them
    lines = list(map(Line, lines))
    linesGrouped = helper.lineGrouper(lines)

    # Draw lines and intersections
    frame_with_lines = frame.copy()
    for index, group in enumerate(linesGrouped):
        color = (255, 0, 255)
        if index == 0:
            color = (255, 0, 0)
        elif index == 1:
            color = (0, 0, 255)

        for line in group:
            cv2.line(frame_with_lines, -line, +line, color, 3)

    intersections = helper.find_field_intersections(linesGrouped, field_mask)
    result_frame = helper.draw_intersections(frame_with_lines, intersections)

    return result_frame


def process_football_videos(clips_folder: str, output_file: str) -> None:
    # Initialize helper
    Help = Helper()

    # Get all mp4 files in the clips folder
    clips_path = Path(clips_folder)
    video_files = sorted(list(clips_path.glob('*.mp4')))

    if not video_files:
        raise ValueError(f"No MP4 files found in {clips_folder}")

    # Get video properties from first file
    first_cap = cv2.VideoCapture(str(video_files[0]))
    if not first_cap.isOpened():
        raise ValueError(f"Error opening first video file: {video_files[0]}")

    frame_width = int(first_cap.get(3))
    frame_height = int(first_cap.get(4))
    fps = int(round(first_cap.get(5)))
    first_cap.release()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process each video file
    for video_file in video_files:
        print(f"Processing {video_file.name}...")
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Warning: Could not open {video_file.name}, skipping...")
            continue

        # Add text showing current clip name
        clip_frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            result_frame = process_football_frame(frame, Help)

            # Add clip name and progress
            clip_frame_count += 1
            progress = (clip_frame_count / total_frames) * 100
            cv2.putText(result_frame, f"Clip: {video_file.name} ({progress:.1f}%)",
                        (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame to output
            out.write(result_frame)

        cap.release()

    # Release resources
    out.release()
    print("Video processing completed.")


def main():
    clips_folder = './media/football_clips'
    output_file = 'football_lines_detected.mp4'

    try:
        process_football_videos(clips_folder, output_file)
    except Exception as e:
        print(f"Error processing videos: {e}")


if __name__ == '__main__':
    main()