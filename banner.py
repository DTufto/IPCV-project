import cv2
import numpy as np
from helper import Helper, Line
from goalpost import GoalpostDetector


class BannerPlacer:
    def __init__(self, banner_image_path):
        self.helper = Helper()
        self.goalpost_detector = GoalpostDetector()
        self.banner = cv2.imread(banner_image_path)
        if self.banner is None:
            raise ValueError("Could not load banner image")

        self.GOALPOST_HEIGHT = 2.44
        self.GOALPOST_WIDTH = 7.32

    def get_banner_points(self, frame, field_mask, goalpost_rect, intersections):
        """Calculate banner placement points using goalpost and field intersections"""
        x, y, w, h = goalpost_rect
        base_point = (x + w // 2, y + h)

        relevant_intersections = []
        max_dist = frame.shape[1] * 0.3

        for intersection in intersections:
            dist = np.sqrt((intersection[0] - base_point[0]) ** 2 +
                           (intersection[1] - base_point[1]) ** 2)
            if dist < max_dist:
                relevant_intersections.append(intersection)

        if len(relevant_intersections) < 2:
            banner_width = w * 5
            banner_height = int(banner_width * 0.2)

            banner_left = base_point[0] - banner_width // 2
            banner_right = base_point[0] + banner_width // 2
            banner_top = base_point[1] - banner_height
            banner_bottom = base_point[1]
        else:
            relevant_intersections.sort(key=lambda p:
            np.sqrt((p[0] - base_point[0]) ** 2 + (p[1] - base_point[1]) ** 2))

            left_point = min(relevant_intersections[:2], key=lambda p: p[0])
            right_point = max(relevant_intersections[:2], key=lambda p: p[0])

            banner_width = int(right_point[0] - left_point[0]) * 2
            banner_height = int(banner_width * 0.2)

            banner_left = base_point[0] - banner_width // 2
            banner_right = base_point[0] + banner_width // 2
            banner_top = base_point[1] - banner_height
            banner_bottom = base_point[1]

        MIN_WIDTH = 100
        MIN_HEIGHT = 20
        if banner_width < MIN_WIDTH:
            scale = MIN_WIDTH / banner_width
            banner_width = MIN_WIDTH
            banner_height = int(banner_height * scale)
        if banner_height < MIN_HEIGHT:
            banner_height = MIN_HEIGHT

        banner_left = max(0, min(banner_left, frame.shape[1]))
        banner_right = max(0, min(banner_right, frame.shape[1]))
        banner_top = max(0, min(banner_top, frame.shape[0]))
        banner_bottom = max(0, min(banner_bottom, frame.shape[0]))

        return np.array([
            [banner_left, banner_bottom],
            [banner_right, banner_bottom],
            [banner_left, banner_top],
            [banner_right, banner_top]
        ], dtype=np.float32)

    def place_banner(self, frame):
        """Place banner on frame using goalpost detection and field intersections"""
        field_mask = self.helper.create_field_mask(frame)
        if field_mask is None:
            return frame

        goalposts, _ = self.goalpost_detector.detect_goalposts(frame, field_mask, self.helper)
        if not goalposts:
            return frame

        goalpost_rect = max(goalposts, key=lambda x: x[3])

        lines = self.helper.detect_lines_on_field(frame, field_mask)
        if lines is None:
            return frame

        lines = list(map(lambda x: Line(x), lines))
        grouped_lines = self.helper.lineGrouper(lines)
        intersections = self.helper.find_field_intersections(grouped_lines, field_mask)

        banner_points = self.get_banner_points(frame, field_mask, goalpost_rect, intersections)
        if banner_points is None:
            return frame

        target_width = int(banner_points[1][0] - banner_points[0][0])
        target_height = int(banner_points[0][1] - banner_points[2][1])

        if target_width <= 0 or target_height <= 0:
            return frame

        try:
            resized_banner = cv2.resize(self.banner, (target_width, target_height))
        except Exception:
            return frame

        h_banner, w_banner = resized_banner.shape[:2]
        src_points = np.array([
            [0, h_banner],
            [w_banner, h_banner],
            [0, 0],
            [w_banner, 0]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_points, banner_points)
        if H is None:
            return frame

        h_img, w_img = frame.shape[:2]
        warped_banner = cv2.warpPerspective(resized_banner, H, (w_img, h_img))

        mask = np.ones_like(resized_banner)
        warped_mask = cv2.warpPerspective(mask, H, (w_img, h_img))

        result = frame.copy()
        result = result * (1 - warped_mask) + warped_banner * warped_mask

        return result.astype(np.uint8)


def main():
    banner_placer = BannerPlacer('media/windows7_whopper.jpg')
    video_path = 'media/football_clips/clip2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    for frame_num in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        result = banner_placer.place_banner(frame)

        if frame_num == 0:
            cv2.imwrite('debug_frame_1.png', result)
        elif frame_num == 9:
            cv2.imwrite('debug_frame_10.png', result)

        cv2.imshow("Banner Placement", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()