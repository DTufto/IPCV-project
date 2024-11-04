import cv2
import numpy as np
from typing import List, Tuple, Optional


class PenaltySpotDetector:
    def __init__(self):
        # FIFA regulation measurements in meters
        self.PENALTY_SPOT_DISTANCE = 11.0  # Distance from goal line
        self.PENALTY_AREA_LENGTH = 16.5  # Length of penalty area
        self.PENALTY_AREA_WIDTH = 40.32  # Width of penalty area
        self.FIELD_WIDTH = 68.0  # Standard field width
        self.FIELD_LENGTH = 105.0  # Standard field length

    def _create_spot_mask(self, gray: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        """Create a mask specifically targeting small, bright, circular spots"""
        # Calculate field statistics for adaptive thresholding
        field_mean = cv2.mean(gray, mask=field_mask)[0]
        field_std = cv2.meanStdDev(gray, mask=field_mask)[1][0][0]

        # Multi-level thresholding for better spot detection
        thresh_low = min(255, int(field_mean + 2.0 * field_std))
        thresh_high = min(255, int(field_mean + 3.0 * field_std))

        # Create binary masks at different thresholds
        mask_low = cv2.threshold(gray, thresh_low, 255, cv2.THRESH_BINARY)[1]
        mask_high = cv2.threshold(gray, thresh_high, 255, cv2.THRESH_BINARY)[1]

        # Use local adaptive thresholding as well
        mask_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Smaller block size for small spots
            -2
        )

        # Combine masks
        combined_mask = cv2.bitwise_and(mask_low, mask_adaptive)
        combined_mask = cv2.bitwise_and(combined_mask, field_mask)

        # Remove long connected components (likely player uniforms)
        return self._remove_non_circular_objects(combined_mask)

    def _remove_non_circular_objects(self, mask: np.ndarray) -> np.ndarray:
        """Remove objects that are likely not penalty spots based on shape"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Create output mask
        result_mask = np.zeros_like(mask)

        # Filter components based on size and shape
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Size constraints for penalty spot
            if not (9 <= area <= 100):  # Tighter area constraints
                continue

            # Aspect ratio check - should be roughly circular
            if max(w, h) / min(w, h) > 1.3:  # Stricter aspect ratio
                continue

            # Compactness check
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.7:  # Higher circularity threshold
                        continue

            # If all checks pass, add to result mask
            result_mask |= component_mask

        return result_mask

    def detect_penalty_spots(self, frame: np.ndarray, field_mask: np.ndarray, white_mask: np.ndarray) -> List[
        Tuple[int, int]]:
        """Detect penalty spots with improved filtering"""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create ROI mask for penalty area
        penalty_area_mask = self._create_penalty_area_mask(field_mask)

        # Create spot-specific mask
        spot_mask = self._create_spot_mask(gray, field_mask)

        # Apply penalty area mask
        spot_search_area = cv2.bitwise_and(spot_mask, penalty_area_mask)

        # Find candidate spots using connected components
        spots = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spot_search_area, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            centroid = (int(centroids[i][0]), int(centroids[i][1]))

            # Additional validation
            if self._validate_spot(centroid, stats[i], gray, field_mask):
                spots.append(centroid)

        return spots

    def _validate_spot(self, center: Tuple[int, int], stats: np.ndarray,
                       gray: np.ndarray, field_mask: np.ndarray) -> bool:
        """Validate a potential penalty spot"""
        x, y, w, h, area = stats

        # Check local intensity pattern
        roi_size = max(w, h) + 4
        x1 = max(0, center[0] - roi_size)
        x2 = min(gray.shape[1], center[0] + roi_size)
        y1 = max(0, center[1] - roi_size)
        y2 = min(gray.shape[0], center[1] + roi_size)

        roi = gray[y1:y2, x1:x2]
        roi_mask = field_mask[y1:y2, x1:x2]

        if roi.size == 0 or roi_mask.size == 0:
            return False

        # Check if spot is brighter than surroundings
        spot_value = gray[center[1], center[0]]
        surroundings = cv2.mean(roi, mask=roi_mask)[0]

        if spot_value < surroundings + 30:  # Spot should be significantly brighter
            return False

        return True

    def _create_penalty_area_mask(self, field_mask: np.ndarray) -> np.ndarray:
        """
        Create a mask for the penalty area using FIFA standard measurements
        """
        h, w = field_mask.shape
        mask = np.zeros_like(field_mask)

        # Get field points
        field_points = np.where(field_mask > 0)
        if len(field_points[0]) == 0:
            return mask

        # Find field boundaries
        top = np.min(field_points[0])
        bottom = np.max(field_points[0])
        left = np.min(field_points[1])
        right = np.max(field_points[1])

        # Determine if we're looking at left or right side of field
        left_half_count = np.sum(field_mask[:, :w // 2] > 0)
        right_half_count = np.sum(field_mask[:, w // 2:] > 0)
        looking_at_left = left_half_count > right_half_count

        # Calculate visible field dimensions
        visible_width = right - left
        visible_height = bottom - top

        # Find the vertical center line of the field
        vertical_profile = np.sum(field_mask, axis=1)
        field_center_y = np.argmax(vertical_profile)

        # Estimate perspective scale factor
        # Assume the visible width at the bottom is approximately half the field width
        bottom_scale = self.FIELD_WIDTH / (2 * visible_width)

        # Create a search region that's larger than strictly necessary
        search_width = int(self.PENALTY_AREA_WIDTH * 1.5 / bottom_scale)  # 50% wider than penalty area
        search_height = int(self.PENALTY_AREA_LENGTH * 1.5 / bottom_scale)  # 50% taller than penalty area

        # Position the search region based on which goal we're looking at
        if looking_at_left:
            # Left goal - penalty spot is 11m from left edge
            spot_x = left + int(self.PENALTY_SPOT_DISTANCE / bottom_scale)
            x1 = max(0, spot_x - search_width // 2)
            x2 = min(w, spot_x + search_width // 2)
        else:
            # Right goal - penalty spot is 11m from right edge
            spot_x = right - int(self.PENALTY_SPOT_DISTANCE / bottom_scale)
            x1 = max(0, spot_x - search_width // 2)
            x2 = min(w, spot_x + search_width // 2)

        # Vertical position - center around the field's vertical center
        y1 = max(0, field_center_y - search_height // 2)
        y2 = min(h, field_center_y + search_height // 2)

        # Create the mask
        mask[y1:y2, x1:x2] = 255

        # AND with field mask to ensure we're only looking at valid field areas
        mask = cv2.bitwise_and(mask, field_mask)

        return mask

    def _evaluate_penalty_spot(self, x: int, y: int, r: int, search_area: np.ndarray) -> float:
        """Evaluate how likely a detected circle is to be the penalty spot"""
        # Get ROI around the circle
        y1 = max(0, y - r)
        y2 = min(search_area.shape[0], y + r)
        x1 = max(0, x - r)
        x2 = min(search_area.shape[1], x + r)
        roi = search_area[y1:y2, x1:x2]

        if roi.size == 0:
            return 0

        # Calculate circularity
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Check isolation (not part of a line)
        y1_large = max(0, y - r * 2)
        y2_large = min(search_area.shape[0], y + r * 2)
        x1_large = max(0, x - r * 2)
        x2_large = min(search_area.shape[1], x + r * 2)
        larger_roi = search_area[y1_large:y2_large, x1_large:x2_large]

        if larger_roi.size == 0:
            return 0

        isolation = 1 - (np.sum(larger_roi > 0) / larger_roi.size)

        # Combine metrics
        return circularity * 0.6 + isolation * 0.4