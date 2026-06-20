import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

from app.backends.base import BackendUnavailable, SegmentationBackend, SegmentationCandidate
from app.backends.opencv_backend import _detect_dark_drawer_mat
from app.postprocess import mask_to_bbox


class RoboflowSam3Backend(SegmentationBackend):
    name = "roboflow_sam3"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        api_key_file: str = "",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.api_key_file = api_key_file
        self.timeout_seconds = timeout_seconds

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        api_key = self._api_key()
        payload = {
            "image": {
                "type": "base64",
                "value": _image_to_base64(image),
            },
            "prompts": [{"type": "text", "text": prompt} for prompt in prompts],
            "output_prob_thresh": 0.25,
            "format": "polygon",
        }
        try:
            response = requests.post(
                f"{self.base_url}/sam3/concept_segment",
                params={"api_key": api_key},
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            message = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
            raise BackendUnavailable(f"Roboflow SAM3 request failed: {exc} {message}") from exc

        candidates = _response_to_candidates(response.json(), image.size)
        return _filter_to_drawer_mat(candidates, image)

    def health(self) -> dict[str, Any]:
        try:
            api_key = self._api_key()
        except BackendUnavailable as exc:
            return {
                "available": False,
                "model_loaded": False,
                "message": str(exc),
            }
        return {
            "available": bool(api_key),
            "model_loaded": False,
            "message": None,
        }

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_file:
            key_path = Path(self.api_key_file).expanduser()
            if not key_path.exists():
                raise BackendUnavailable(f"ROBOFLOW_API_KEY_FILE does not exist: {key_path}")
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                return key
        raise BackendUnavailable(
            "Set ROBOFLOW_API_KEY or ROBOFLOW_API_KEY_FILE to run Roboflow SAM3."
        )


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _response_to_candidates(
    payload: dict[str, Any],
    image_size: tuple[int, int],
) -> list[SegmentationCandidate]:
    candidates: list[SegmentationCandidate] = []
    for prompt_result in payload.get("prompt_results", []):
        echo = prompt_result.get("echo", {})
        label = str(echo.get("text") or echo.get("class_name") or "tool")
        for prediction in prompt_result.get("predictions", []):
            score = float(prediction.get("confidence") or prediction.get("score") or 0.0)
            for polygon in _prediction_polygons(prediction):
                mask = _polygon_to_mask(polygon, image_size)
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    continue
                candidates.append(
                    SegmentationCandidate(
                        label=label,
                        source_prompt=label,
                        score=score,
                        bbox_xyxy=bbox,
                        mask=mask,
                    )
                )
    return candidates


def _prediction_polygons(prediction: dict[str, Any]) -> list[list[tuple[int, int]]]:
    masks = prediction.get("masks") or prediction.get("polygons") or []
    polygons: list[list[tuple[int, int]]] = []
    for mask in masks:
        if not mask:
            continue
        if isinstance(mask, dict):
            points = mask.get("points") or mask.get("polygon") or []
        else:
            points = mask
        polygon = [
            (int(round(point[0])), int(round(point[1])))
            for point in points
            if len(point) >= 2
        ]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _polygon_to_mask(polygon: list[tuple[int, int]], image_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)
    return np.array(mask_image, dtype=bool)


def _filter_to_drawer_mat(
    candidates: list[SegmentationCandidate],
    image: Image.Image,
    min_overlap_ratio: float = 0.50,
) -> list[SegmentationCandidate]:
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mat_mask = _detect_dark_drawer_mat(hsv).astype(bool)
    filtered: list[SegmentationCandidate] = []
    for candidate in candidates:
        area = np.count_nonzero(candidate.mask)
        if area == 0:
            continue
        overlap_ratio = np.count_nonzero(candidate.mask & mat_mask) / area
        if overlap_ratio >= min_overlap_ratio:
            filtered.append(candidate)
    return filtered
