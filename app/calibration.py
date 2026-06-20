from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class BoardCalibration:
    board_size_mm: float
    corners_px: list[tuple[float, float]]
    homography_px_to_mm: list[list[float]]

    def point_to_mm(self, point: tuple[int, int] | tuple[float, float]) -> tuple[float, float]:
        x, y = point
        vector = np.array([x, y, 1.0], dtype=np.float64)
        matrix = np.array(self.homography_px_to_mm, dtype=np.float64)
        mapped = matrix @ vector
        mapped /= mapped[2]
        return round(float(mapped[0]), 2), round(float(mapped[1]), 2)

    def bbox_to_mm(
        self,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        corners = [
            self.point_to_mm((x1, y1)),
            self.point_to_mm((x2, y1)),
            self.point_to_mm((x2, y2)),
            self.point_to_mm((x1, y2)),
        ]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        return round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)


def detect_light_square_board(
    image: Image.Image,
    board_size_mm: float,
) -> BoardCalibration | None:
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    light = ((value > 145) & (saturation < 85)).astype(np.uint8) * 255
    light = cv2.morphologyEx(
        light,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=2,
    )
    light = cv2.morphologyEx(
        light,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=1,
    )
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.size[0] * image.size[1]
    candidates = [contour for contour in contours if cv2.contourArea(contour) > image_area * 0.20]
    if not candidates:
        return None

    contour = max(candidates, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered = _order_corners(box)
    destination = np.array(
        [
            [0.0, 0.0],
            [board_size_mm, 0.0],
            [board_size_mm, board_size_mm],
            [0.0, board_size_mm],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(ordered, destination)
    return BoardCalibration(
        board_size_mm=board_size_mm,
        corners_px=[(round(float(x), 2), round(float(y), 2)) for x, y in ordered],
        homography_px_to_mm=homography.round(8).tolist(),
    )


def mask_overlap_with_board(mask: np.ndarray, calibration: BoardCalibration) -> float:
    height, width = mask.shape
    board_mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(calibration.corners_px, dtype=np.int32)
    cv2.fillConvexPoly(board_mask, points, 1)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask & board_mask.astype(bool)) / area)


def _order_corners(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered

