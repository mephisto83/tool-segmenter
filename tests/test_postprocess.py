import numpy as np
from PIL import Image

from app.postprocess import (
    box_iou,
    mask_area,
    mask_centroid,
    mask_iou,
    mask_to_bbox,
    mask_to_contours,
    resize_image_for_model,
)


def test_resize_image_for_model_does_not_upscale() -> None:
    image = Image.new("RGB", (100, 80))
    resized, metadata = resize_image_for_model(image, 200)
    assert resized.size == (100, 80)
    assert metadata.scale == 1.0


def test_resize_image_for_model_scales_long_side() -> None:
    image = Image.new("RGB", (400, 200))
    resized, metadata = resize_image_for_model(image, 100)
    assert resized.size == (100, 50)
    assert metadata.scale == 0.25


def test_mask_to_bbox_area_and_centroid() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:10, 8:18] = True
    assert mask_to_bbox(mask) == (8, 4, 18, 10)
    assert mask_area(mask) == 60
    assert mask_centroid(mask) == (12, 6)


def test_mask_to_contours_returns_outline_and_hole() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 15:25] = False
    outline, holes = mask_to_contours(mask, epsilon_ratio=0.01)
    assert len(outline) >= 4
    assert len(holes) == 1
    assert len(holes[0]) >= 4


def test_box_and_mask_iou() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175

    mask_a = np.zeros((10, 10), dtype=bool)
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_a[:5, :5] = True
    mask_b[2:7, 2:7] = True
    assert round(mask_iou(mask_a, mask_b), 4) == round(9 / 41, 4)

