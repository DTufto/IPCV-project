import cv2
from helper import Helper, Line
from pathlib import Path

def process_football_frame(frame, helper):
    field_mask = helper.create_field_mask(frame)
    lines = helper.detect_lines_on_field(frame, field_mask)

    if lines is None:
        return frame

    lines = list(map(Line, lines))
    linesGrouped = helper.lineGrouper(lines)

    frame_with_lines = frame.copy()
    for group in linesGrouped:
        color = (255, 0, 255)
        for line in group:
            cv2.line(frame_with_lines, -line, +line, color, 3)

    intersections = helper.find_field_intersections(linesGrouped, field_mask)
    result_frame = helper.draw_intersections(frame_with_lines, intersections)

    return result_frame

def process_football_videos(clips_folder: str, output_file: str) -> None:
    Help = Helper()
    clips_path = Path(clips_folder)
    video_files = sorted(list(clips_path.glob('*.mp4')))

    if not video_files:
        raise ValueError(f"No MP4 files found in {clips_folder}")

    first_cap = cv2.VideoCapture(str(video_files[0]))
    if not first_cap.isOpened():
        raise ValueError(f"Error opening first video file: {video_files[0]}")

    frame_width = int(first_cap.get(3))
    frame_height = int(first_cap.get(4))
    fps = int(round(first_cap.get(5)))
    first_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            continue

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_frame = process_football_frame(frame, Help)
            out.write(result_frame)

        cap.release()

    out.release()

process_football_videos('./media/football_clips', 'outputtie.mp4')