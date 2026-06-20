from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.backends.base import SegmentationBackend, SegmentationCandidate
from app.postprocess import mask_to_bbox


@dataclass(slots=True)
class _SeedComponent:
    bbox: tuple[int, int, int, int]
    area: int
    label_id: int


class OpenCvToolBackend(SegmentationBackend):
    name = "opencv"

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        del prompts
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mat_mask = _detect_dark_drawer_mat(hsv)
        mat_bbox = mask_to_bbox(mat_mask.astype(bool)) or (0, 0, image.size[0], image.size[1])
        seed_mask = _build_foreground_seed(hsv, mat_mask)
        components, labels = _seed_components(seed_mask, mat_mask)
        groups = _group_components(components)

        candidates: list[SegmentationCandidate] = []
        for group in groups:
            mask = _group_to_mask(group, labels, image.size[::-1])
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            label = _infer_label(bbox)
            refinement_bbox = expand_bbox_for_refinement(
                bbox,
                image_size=image.size,
                clip_bbox=mat_bbox,
            )
            candidates.append(
                SegmentationCandidate(
                    label=label,
                    source_prompt="opencv_foreground_segmentation",
                    score=_score_group(group, bbox),
                    bbox_xyxy=bbox,
                    mask=mask,
                    refinement_bbox_xyxy=refinement_bbox,
                )
            )
        return candidates


def expand_bbox_for_refinement(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    clip_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Expand a partial evidence box into the crop/box SAM-style refinement should see."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = max(18, round(width * 0.28))
    pad_y = max(18, round(height * 0.18))

    if height > width * 2.2:
        pad_x = max(pad_x, round(width * 0.75), 34)
        pad_y = max(pad_y, round(height * 0.16), 42)
    elif width > height * 2.2:
        pad_x = max(pad_x, round(width * 0.15), 48)
        pad_y = max(pad_y, round(height * 1.25), 34)
    else:
        pad_x = max(pad_x, round(width * 0.35))
        pad_y = max(pad_y, round(height * 0.35))

    if min(width, height) < 55 and max(width, height) > 110:
        if height >= width:
            pad_x = max(pad_x, 46)
            pad_y = max(pad_y, round(height * 0.22))
        else:
            pad_x = max(pad_x, round(width * 0.18))
            pad_y = max(pad_y, 46)

    expanded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    return _clip_box(expanded, clip_bbox or (0, 0, image_width, image_height))


def _detect_dark_drawer_mat(hsv: np.ndarray) -> np.ndarray:
    height, width = hsv.shape[:2]
    value = hsv[:, :, 2]
    dark = (value < 85).astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [contour for contour in contours if cv2.contourArea(contour) > width * height * 0.05]
    if not usable:
        return np.ones((height, width), dtype=np.uint8) * 255

    contour = max(usable, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(contour)
    roi = value[y : y + box_height, x : x + box_width] < 85
    row_fraction = roi.mean(axis=1)
    col_fraction = roi.mean(axis=0)
    row_indices = np.where(row_fraction > 0.55)[0]
    col_indices = np.where(col_fraction > 0.45)[0]

    top = y + int(row_indices[0]) if row_indices.size else y
    bottom = y + int(row_indices[-1]) if row_indices.size else y + box_height
    left = x + int(col_indices[0]) if col_indices.size else x
    right = x + int(col_indices[-1]) if col_indices.size else x + box_width

    # Trim bright drawer rails and lips that are adjacent to, but not part of, the work surface.
    x_margin = max(20, round((right - left) * 0.04))
    y_margin = max(10, round((bottom - top) * 0.004))
    left = max(0, left + x_margin)
    right = min(width, right - x_margin)
    top = max(0, top + y_margin)
    bottom = min(height, bottom - y_margin)

    mat_mask = np.zeros((height, width), dtype=np.uint8)
    if right > left and bottom > top:
        mat_mask[top:bottom, left:right] = 255
    else:
        mat_mask[y : y + box_height, x : x + box_width] = 255
    return mat_mask


def _build_foreground_seed(hsv: np.ndarray, mat_mask: np.ndarray) -> np.ndarray:
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = ((saturation > 55) & (value > 50)).astype(np.uint8) * 255
    metal_or_light = ((value > 145) & (saturation < 115)).astype(np.uint8) * 255
    seed = cv2.bitwise_or(colored, metal_or_light)
    seed = cv2.bitwise_and(seed, mat_mask)
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return seed


def _seed_components(
    seed_mask: np.ndarray,
    mat_mask: np.ndarray,
) -> tuple[list[_SeedComponent], np.ndarray]:
    mat_bbox = mask_to_bbox(mat_mask.astype(bool))
    if mat_bbox is None:
        mat_bbox = (0, 0, seed_mask.shape[1], seed_mask.shape[0])
    mat_x1, mat_y1, mat_x2, mat_y2 = mat_bbox
    mat_width = mat_x2 - mat_x1
    mat_height = mat_y2 - mat_y1

    _, labels, stats, _ = cv2.connectedComponentsWithStats((seed_mask > 0).astype(np.uint8), 8)
    components: list[_SeedComponent] = []
    for label_id in range(1, stats.shape[0]):
        x, y, width, height, area = [int(value) for value in stats[label_id]]
        if area < 550 or width < 8 or height < 8:
            continue
        if y < mat_y1 + 45 and height < 45:
            continue
        touches_vertical_rail = (x <= mat_x1 + 8 or x + width >= mat_x2 - 8) and (
            height > 0.18 * mat_height
        )
        touches_horizontal_lip = (y <= mat_y1 + 8 or y + height >= mat_y2 - 8) and (
            width > 0.25 * mat_width
        )
        if touches_vertical_rail or touches_horizontal_lip:
            continue
        components.append(
            _SeedComponent(
                bbox=(x, y, x + width, y + height),
                area=area,
                label_id=label_id,
            )
        )
    return components, labels


def _group_components(components: list[_SeedComponent]) -> list[list[_SeedComponent]]:
    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(components):
        for right_index in range(left_index + 1, len(components)):
            right = components[right_index]
            if _should_group(left.bbox, right.bbox):
                union(left_index, right_index)

    groups: dict[int, list[_SeedComponent]] = {}
    for index, component in enumerate(components):
        groups.setdefault(find(index), []).append(component)

    filtered: list[list[_SeedComponent]] = []
    for group in groups.values():
        x1, y1, x2, y2 = _union_bbox(group)
        area = sum(component.area for component in group)
        if area < 750 and max(x2 - x1, y2 - y1) < 80:
            continue
        filtered.append(group)
    return sorted(filtered, key=lambda group: (_union_bbox(group)[1], _union_bbox(group)[0]))


def _should_group(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    left_width = left[2] - left[0]
    left_height = left[3] - left[1]
    right_width = right[2] - right[0]
    right_height = right[3] - right[1]
    left_center_x = (left[0] + left[2]) / 2
    right_center_x = (right[0] + right[2]) / 2
    left_center_y = (left[1] + left[3]) / 2
    right_center_y = (right[1] + right[3]) / 2

    vertical_neighbors = (
        _is_verticalish(left) or _is_verticalish(right)
    ) and _axis_gap(left, right, axis="y") <= 150
    if vertical_neighbors:
        if _axis_overlap(left, right, axis="x") >= 0.18 * min(left_width, right_width):
            return True
        if abs(left_center_x - right_center_x) <= 55:
            return True

    horizontal_neighbors = (
        _is_horizontalish(left)
        and _is_horizontalish(right)
        and _axis_gap(left, right, axis="x") <= 90
    )
    if horizontal_neighbors:
        if _axis_overlap(left, right, axis="y") >= 0.20 * min(left_height, right_height):
            return True
        if abs(left_center_y - right_center_y) <= 25:
            return True
    return False


def _group_to_mask(
    group: list[_SeedComponent],
    labels: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    seed = np.zeros((height, width), dtype=np.uint8)
    points: list[np.ndarray] = []
    for component in group:
        component_mask = labels == component.label_id
        seed[component_mask] = 255
        ys, xs = np.where(component_mask)
        if xs.size:
            points.append(np.column_stack([xs, ys]).astype(np.int32))

    if not points:
        return seed.astype(bool)

    all_points = np.vstack(points)
    hull = cv2.convexHull(all_points)
    mask = np.zeros_like(seed)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    return mask.astype(bool)


def _infer_label(bbox: tuple[int, int, int, int]) -> str:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect = max(width / max(height, 1), height / max(width, 1))
    if width > height * 4:
        return "tool bit"
    if height > width * 3:
        return "screwdriver"
    if aspect < 1.5 and max(width, height) > 250:
        return "hand tool"
    return "detected tool"


def _score_group(group: list[_SeedComponent], bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    seed_area = sum(component.area for component in group)
    fill_ratio = min(1.0, seed_area / box_area)
    score = 0.45 + 0.35 * fill_ratio + min(0.15, len(group) * 0.03)
    return round(min(0.95, score), 4)


def _union_bbox(group: list[_SeedComponent]) -> tuple[int, int, int, int]:
    return (
        min(component.bbox[0] for component in group),
        min(component.bbox[1] for component in group),
        max(component.bbox[2] for component in group),
        max(component.bbox[3] for component in group),
    )


def _is_verticalish(box: tuple[int, int, int, int]) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return height > width * 1.15


def _is_horizontalish(box: tuple[int, int, int, int]) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width > height * 2.0


def _axis_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    axis: str,
) -> int:
    if axis == "x":
        return max(0, min(left[2], right[2]) - max(left[0], right[0]))
    return max(0, min(left[3], right[3]) - max(left[1], right[1]))


def _axis_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    axis: str,
) -> int:
    if axis == "x":
        return max(0, max(left[0], right[0]) - min(left[2], right[2]))
    return max(0, max(left[1], right[1]) - min(left[3], right[3]))


def _clip_box(
    box: tuple[int, int, int, int],
    clip_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    clip_x1, clip_y1, clip_x2, clip_y2 = clip_bbox
    clipped_x1 = max(clip_x1, min(clip_x2, x1))
    clipped_y1 = max(clip_y1, min(clip_y2, y1))
    clipped_x2 = max(clipped_x1 + 1, min(clip_x2, x2))
    clipped_y2 = max(clipped_y1 + 1, min(clip_y2, y2))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2
