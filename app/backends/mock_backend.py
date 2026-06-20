import numpy as np
from PIL import Image

from app.backends.base import SegmentationBackend, SegmentationCandidate
from app.postprocess import mask_to_bbox


class MockSegmentationBackend(SegmentationBackend):
    name = "mock"

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        width, height = image.size
        prompts = prompts or ["screwdriver", "drill bit", "pliers"]
        candidates: list[SegmentationCandidate] = []

        for prompt in prompts:
            normalized = prompt.lower()
            if "screwdriver" in normalized or "hand tool" in normalized or "handle" in normalized:
                label = "blue handle screwdriver" if "blue" in normalized else "screwdriver"
                score = 0.88 if label != "screwdriver" else 0.84
                candidates.append(
                    _candidate(
                        label,
                        prompt,
                        score,
                        _rect_mask(width, height, 0.16, 0.14, 0.26, 0.78),
                    )
                )
            elif "bit" in normalized:
                label = "drill bit" if "drill" in normalized else "screwdriver bit"
                candidates.append(
                    _candidate(
                        label,
                        prompt,
                        0.78,
                        _rect_mask(width, height, 0.58, 0.20, 0.64, 0.36),
                    )
                )
            elif "plier" in normalized:
                candidates.append(
                    _candidate(
                        "pliers",
                        prompt,
                        0.81,
                        _ellipse_mask(width, height, 0.56, 0.56, 0.82, 0.82),
                    )
                )
            elif "scissor" in normalized:
                candidates.append(
                    _candidate(
                        "scissors",
                        prompt,
                        0.76,
                        _rect_mask(width, height, 0.28, 0.58, 0.48, 0.82),
                    )
                )

        if not candidates:
            candidates.append(
                _candidate(
                    "hand tool",
                    prompts[0],
                    0.65,
                    _rect_mask(width, height, 0.20, 0.20, 0.46, 0.72),
                )
            )
        return candidates


def _candidate(label: str, prompt: str, score: float, mask: np.ndarray) -> SegmentationCandidate:
    bbox = mask_to_bbox(mask)
    if bbox is None:
        raise ValueError("mock candidate produced an empty mask")
    return SegmentationCandidate(
        label=label,
        source_prompt=prompt,
        score=score,
        bbox_xyxy=bbox,
        mask=mask,
    )


def _rect_mask(
    width: int,
    height: int,
    x1_ratio: float,
    y1_ratio: float,
    x2_ratio: float,
    y2_ratio: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x1, y1 = int(width * x1_ratio), int(height * y1_ratio)
    x2, y2 = max(x1 + 1, int(width * x2_ratio)), max(y1 + 1, int(height * y2_ratio))
    mask[y1:y2, x1:x2] = True
    return mask


def _ellipse_mask(
    width: int,
    height: int,
    x1_ratio: float,
    y1_ratio: float,
    x2_ratio: float,
    y2_ratio: float,
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cx = width * (x1_ratio + x2_ratio) / 2
    cy = height * (y1_ratio + y2_ratio) / 2
    rx = max(1.0, width * (x2_ratio - x1_ratio) / 2)
    ry = max(1.0, height * (y2_ratio - y1_ratio) / 2)
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
