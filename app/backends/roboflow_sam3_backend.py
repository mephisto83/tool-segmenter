from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

from app.backends.base import BackendUnavailable, SegmentationBackend, SegmentationCandidate
from app.postprocess import mask_to_bbox


class RoboflowSam3Backend(SegmentationBackend):
    name = "roboflow_sam3"

    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.post(
                f"{self.base_url}/segment",
                headers=headers,
                files={"image": ("image.png", buffer, "image/png")},
                data={"prompts": ",".join(prompts)},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BackendUnavailable(f"Roboflow SAM3 request failed: {exc}") from exc

        payload = response.json()
        predictions = payload.get("predictions") or payload.get("objects") or []
        return [_prediction_to_candidate(prediction, image.size) for prediction in predictions]

    def health(self) -> dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            available = response.ok
            message = None if response.ok else response.text[:200]
        except requests.RequestException as exc:
            available = False
            message = str(exc)
        return {
            "available": available,
            "model_loaded": available,
            "message": message,
        }


def _prediction_to_candidate(
    prediction: dict[str, Any],
    image_size: tuple[int, int],
) -> SegmentationCandidate:
    width, height = image_size
    label = str(prediction.get("label") or prediction.get("class") or "tool")
    source_prompt = str(prediction.get("source_prompt") or prediction.get("prompt") or label)
    score = float(prediction.get("score") or prediction.get("confidence") or 0.0)
    mask = _mask_from_prediction(prediction, width, height)
    bbox = mask_to_bbox(mask) or _bbox_from_prediction(prediction, width, height)
    return SegmentationCandidate(
        label=label,
        source_prompt=source_prompt,
        score=score,
        bbox_xyxy=bbox,
        mask=mask,
    )


def _mask_from_prediction(prediction: dict[str, Any], width: int, height: int) -> np.ndarray:
    polygon = prediction.get("polygon") or prediction.get("points")
    if polygon:
        mask_image = Image.new("L", (width, height), 0)
        points = [(int(point[0]), int(point[1])) for point in polygon]
        ImageDraw.Draw(mask_image).polygon(points, outline=1, fill=1)
        return np.array(mask_image, dtype=bool)

    bbox = _bbox_from_prediction(prediction, width, height)
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = True
    return mask


def _bbox_from_prediction(
    prediction: dict[str, Any],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if "bbox_xyxy" in prediction:
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
    elif "bbox" in prediction:
        bbox = prediction["bbox"]
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            raise BackendUnavailable(f"Unsupported bbox shape from Roboflow response: {bbox}")
    else:
        x_center = float(prediction.get("x", width / 2))
        y_center = float(prediction.get("y", height / 2))
        box_width = float(prediction.get("width", width))
        box_height = float(prediction.get("height", height))
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )

