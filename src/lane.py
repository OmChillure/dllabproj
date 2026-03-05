import cv2
import numpy as np


def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def draw_lane_polygon(
    frame: np.ndarray,
    left_line: tuple[int, int, int, int],
    right_line: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    overlay = np.zeros_like(frame)
    poly_points = np.array(
        [[
            (left_line[0], left_line[1]),
            (left_line[2], left_line[3]),
            (right_line[2], right_line[3]),
            (right_line[0], right_line[1]),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, poly_points, color)
    return cv2.addWeighted(frame, 0.8, overlay, 0.5, 0.0)


def detect_lanes(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    roi_vertices = np.array(
        [[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cropped = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(
        cropped,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        minLineLength=40,
        maxLineGap=25,
    )

    if lines is None:
        return frame

    left_x: list[int] = []
    left_y: list[int] = []
    right_x: list[int] = []
    right_y: list[int] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
        else:
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])

    min_y = int(height * 3 / 5)
    max_y = height

    left_start = left_end = 0
    right_start = right_end = 0

    if left_x and left_y:
        left_fit = np.poly1d(np.polyfit(left_y, left_x, deg=1))
        left_start = int(left_fit(max_y))
        left_end = int(left_fit(min_y))

    if right_x and right_y:
        right_fit = np.poly1d(np.polyfit(right_y, right_x, deg=1))
        right_start = int(right_fit(max_y))
        right_end = int(right_fit(min_y))

    return draw_lane_polygon(
        frame,
        (left_start, max_y, left_end, min_y),
        (right_start, max_y, right_end, min_y),
    )

