from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.backends.base import SegmentationCandidate
from app.schemas import ToolObject


@dataclass(frozen=True, slots=True)
class ResizeMetadata:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float


def resize_image_for_model(image: Image.Image, max_side: int) -> tuple[Image.Image, ResizeMetadata]:
    width, height = image.size
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return image.copy(), ResizeMetadata(width, height, width, height, 1.0)

    scale = max_side / longest
    resized_size = (round(width * scale), round(height * scale))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized, ResizeMetadata(width, height, resized_size[0], resized_size[1], scale)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = mask.astype(bool, copy=False)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def mask_to_contours(
    mask: np.ndarray,
    epsilon_ratio: float,
    min_contour_area: float = 4.0,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    binary = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []

    hierarchy_rows = hierarchy[0] if hierarchy is not None else []
    external_indices = [
        index
        for index, info in enumerate(hierarchy_rows)
        if len(info) >= 4 and info[3] == -1 and cv2.contourArea(contours[index]) >= min_contour_area
    ]
    if not external_indices:
        return [], []

    largest_external = max(external_indices, key=lambda index: cv2.contourArea(contours[index]))
    outline = _approximate_contour(contours[largest_external], epsilon_ratio)

    holes: list[list[tuple[int, int]]] = []
    child_index = int(hierarchy_rows[largest_external][2])
    while child_index != -1:
        if cv2.contourArea(contours[child_index]) >= min_contour_area:
            holes.append(_approximate_contour(contours[child_index], epsilon_ratio))
        child_index = int(hierarchy_rows[child_index][0])

    return outline, holes


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    mask = mask.astype(bool, copy=False)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection = inter_width * inter_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask IoU requires masks with the same shape")
    a = mask_a.astype(bool, copy=False)
    b = mask_b.astype(bool, copy=False)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def dedupe_candidates(
    candidates: Iterable[SegmentationCandidate],
    mask_iou_threshold: float = 0.70,
    box_iou_threshold: float = 0.75,
    score_close_delta: float = 0.05,
) -> list[SegmentationCandidate]:
    kept: list[SegmentationCandidate] = []
    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    for candidate in sorted_candidates:
        duplicate_index = _find_duplicate_index(
            candidate,
            kept,
            mask_iou_threshold=mask_iou_threshold,
            box_iou_threshold=box_iou_threshold,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue

        current = kept[duplicate_index]
        scores_are_close = abs(candidate.score - current.score) <= score_close_delta
        if scores_are_close and _specificity(candidate.label) > _specificity(current.label):
            kept[duplicate_index] = candidate

    return kept


def filter_candidates(
    candidates: Iterable[SegmentationCandidate],
    min_score: float,
    min_area_px: int = 16,
    border_area_ratio_threshold: float = 0.35,
    high_confidence_score: float = 0.75,
) -> list[SegmentationCandidate]:
    filtered: list[SegmentationCandidate] = []
    for candidate in candidates:
        area = mask_area(candidate.mask)
        if candidate.score < min_score or area < min_area_px:
            continue
        if _border_area_ratio(candidate.mask) > border_area_ratio_threshold:
            if candidate.score < high_confidence_score:
                continue
        filtered.append(candidate)
    return filtered


def candidates_to_response_objects(
    candidates: Iterable[SegmentationCandidate],
    image_size: tuple[int, int],
    epsilon_ratio: float,
    return_masks: bool = False,
    return_contours: bool = True,
) -> list[ToolObject]:
    width, height = image_size
    objects: list[ToolObject] = []
    for index, candidate in enumerate(candidates, start=1):
        bbox = mask_to_bbox(candidate.mask) or candidate.bbox_xyxy
        xmin, ymin, xmax, ymax = bbox
        centroid = mask_centroid(candidate.mask) or ((xmin + xmax) // 2, (ymin + ymax) // 2)
        outline, holes = (
            mask_to_contours(candidate.mask, epsilon_ratio) if return_contours else ([], [])
        )
        if not outline:
            outline = _box_outline(bbox)

        objects.append(
            ToolObject(
                id=f"obj_{index:04d}",
                label=candidate.label,
                source_prompt=candidate.source_prompt,
                score=round(float(candidate.score), 4),
                bbox_xyxy=bbox,
                bbox_xywh=(xmin, ymin, xmax - xmin, ymax - ymin),
                bbox_normalized_xyxy=(
                    round(xmin / width, 4),
                    round(ymin / height, 4),
                    round(xmax / width, 4),
                    round(ymax / height, 4),
                ),
                centroid_xy=centroid,
                area_px=mask_area(candidate.mask),
                outline=outline,
                holes=holes,
                mask_rle=_encode_binary_rle(candidate.mask) if return_masks else None,
            )
        )
    return objects


def _approximate_contour(contour: np.ndarray, epsilon_ratio: float) -> list[tuple[int, int]]:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.5, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)
    return [(int(x), int(y)) for x, y in points]


def _find_duplicate_index(
    candidate: SegmentationCandidate,
    kept: list[SegmentationCandidate],
    mask_iou_threshold: float,
    box_iou_threshold: float,
) -> int | None:
    for index, existing in enumerate(kept):
        same_mask_shape = candidate.mask.shape == existing.mask.shape
        overlaps_by_mask = (
            same_mask_shape and mask_iou(candidate.mask, existing.mask) >= mask_iou_threshold
        )
        overlaps_by_box = box_iou(candidate.bbox_xyxy, existing.bbox_xyxy) >= box_iou_threshold
        if overlaps_by_mask or overlaps_by_box:
            return index
    return None


def _specificity(label: str) -> int:
    generic_penalty = 2 if label.strip().lower() in {"tool", "hand tool", "object"} else 0
    return len(label.split()) - generic_penalty


def _border_area_ratio(mask: np.ndarray) -> float:
    area = mask_area(mask)
    if area == 0:
        return 0.0
    border_pixels = (
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    return float(border_pixels / area)


def _box_outline(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    xmin, ymin, xmax, ymax = box
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def _encode_binary_rle(mask: np.ndarray) -> list[int]:
    flat = mask.astype(bool, copy=False).ravel(order="C")
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        if bool(value) == current:
            run += 1
        else:
            counts.append(run)
            current = bool(value)
            run = 1
    counts.append(run)
    return counts
