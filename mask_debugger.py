import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import Helper
import os


def detect_players(img, hsv):
    """
    Detect players using multiple techniques
    """
    h, s, v = cv2.split(hsv)

    # 1. Jersey color detection
    # Blue team
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

    # Red team
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Referee (black/dark colors)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    # 2. Local variance for detecting detailed areas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    _, variance_mask = cv2.threshold(
        cv2.GaussianBlur(cv2.Laplacian(gray, cv2.CV_8U), (5, 5), 0),
        10, 255, cv2.THRESH_BINARY
    )

    # Combine all player detection methods
    player_mask = cv2.bitwise_or(blue_mask, red_mask)
    player_mask = cv2.bitwise_or(player_mask, black_mask)
    player_mask = cv2.bitwise_or(player_mask, variance_mask)

    # Clean up player mask
    kernel = np.ones((5, 5), np.uint8)
    player_mask = cv2.dilate(player_mask, kernel, iterations=2)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)

    return player_mask


def create_grass_mask(hsv, debug_folder=None):
    """
    Create a robust grass mask that accounts for color variations
    """
    h, s, v = cv2.split(hsv)

    # Multiple grass color ranges
    grass_masks = []

    # Main grass color
    grass_masks.append(cv2.inRange(hsv, np.array([35, 40, 30]), np.array([85, 255, 200])))

    # Lighter grass / chalk lines
    grass_masks.append(cv2.inRange(hsv, np.array([35, 20, 150]), np.array([85, 100, 255])))

    # Darker grass shadows
    grass_masks.append(cv2.inRange(hsv, np.array([35, 40, 20]), np.array([85, 255, 150])))

    # Combine all grass masks
    combined_mask = grass_masks[0]
    for mask in grass_masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    if debug_folder:
        for i, mask in enumerate(grass_masks):
            cv2.imwrite(f'{debug_folder}4a_grass_mask_{i}.jpg', mask)
        cv2.imwrite(f'{debug_folder}4b_combined_grass_mask.jpg', combined_mask)

    return combined_mask


def analyze_field_mask(img, debug_folder='debug_masks/'):
    """
    Enhanced field mask analysis with improved grass detection and player filtering
    """
    os.makedirs(debug_folder, exist_ok=True)

    # Save original
    cv2.imwrite(f'{debug_folder}0_original.jpg', img)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Save HSV channels
    cv2.imwrite(f'{debug_folder}1_hue_channel.jpg', h)
    cv2.imwrite(f'{debug_folder}2_saturation_channel.jpg', s)
    cv2.imwrite(f'{debug_folder}3_value_channel.jpg', v)

    # Create initial field mask
    field_mask = create_grass_mask(hsv, debug_folder)

    # Detect players
    player_mask = detect_players(img, hsv)
    cv2.imwrite(f'{debug_folder}5_player_detection.jpg', player_mask)

    # Initial morphological operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((7, 7), np.uint8)
    kernel_large = np.ones((21, 21), np.uint8)

    # Fill small gaps (like lines)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_medium)

    # Find the main field area using contours
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour (the field)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask from largest contour
        field_mask = np.zeros_like(field_mask)
        cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)

        # Fill any holes in the field mask
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)

    cv2.imwrite(f'{debug_folder}6_initial_field_mask.jpg', field_mask)

    # Remove players from field mask
    field_mask[player_mask > 0] = 0
    cv2.imwrite(f'{debug_folder}7_field_without_players.jpg', field_mask)

    # Fill gaps created by player removal
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_large)
    cv2.imwrite(f'{debug_folder}8_final_mask.jpg', field_mask)

    # Apply mask to original image
    masked_result = cv2.bitwise_and(img, img, mask=field_mask)
    cv2.imwrite(f'{debug_folder}9_masked_result.jpg', masked_result)

    # Create visualization overlay
    overlay = img.copy()
    overlay[field_mask > 0] = (0, 255, 0)
    alpha = 0.3
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(f'{debug_folder}10_mask_overlay.jpg', blended)

    # Show player detections on final result
    player_overlay = blended.copy()
    player_overlay[player_mask > 0] = (0, 0, 255)
    cv2.imwrite(f'{debug_folder}11_player_overlay.jpg', player_overlay)

    return field_mask


def test_improved_mask():
    # Read a sample frame
    cap = cv2.VideoCapture('./media/football_clips/clip1.mp4')
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Analyze and create improved mask
        improved_mask = analyze_field_mask(frame)

        # Detect lines with improved mask
        helper = Helper()
        lines = helper.detect_lines_on_field(frame, improved_mask)

        # Draw detected lines and save result
        if lines is not None:
            result_frame = frame.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite('debug_masks/12_final_line_detection.jpg', result_frame)


if __name__ == "__main__":
    test_improved_mask()