import numpy as np
from PIL import Image, ImageDraw

from app.backends.base import SegmentationCandidate
from app.backends.roboflow_sam3_backend import _filter_candidates
from app.calibration import detect_light_square_board


def test_detect_light_square_board_maps_corners_to_mm() -> None:
    image = Image.new("RGB", (300, 300), (20, 25, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 250, 250), fill=(235, 235, 235))

    calibration = detect_light_square_board(image, 556)

    assert calibration is not None
    assert calibration.board_size_mm == 556
    assert calibration.point_to_mm(calibration.corners_px[0]) == (0.0, 0.0)
    assert calibration.point_to_mm(calibration.corners_px[2]) == (556.0, 556.0)


def test_auto_filter_uses_detected_light_board() -> None:
    image = Image.new("RGB", (300, 300), (20, 25, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 250, 250), fill=(235, 235, 235))

    inside_mask = np.zeros((300, 300), dtype=bool)
    inside_mask[90:130, 90:130] = True
    outside_mask = np.zeros((300, 300), dtype=bool)
    outside_mask[10:40, 10:40] = True
    inside = SegmentationCandidate("tool", "tool", 0.9, (90, 90, 129, 129), inside_mask)
    outside = SegmentationCandidate("tool", "tool", 0.9, (10, 10, 39, 39), outside_mask)

    filtered = _filter_candidates([inside, outside], image, "auto", 556)

    assert len(filtered) == 1
    assert filtered[0] is inside
