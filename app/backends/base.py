from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


class BackendUnavailable(RuntimeError):
    """Raised when a selected segmentation backend cannot serve a request."""


@dataclass(slots=True)
class SegmentationCandidate:
    label: str
    source_prompt: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray

    def __post_init__(self) -> None:
        self.mask = self.mask.astype(bool, copy=False)


class SegmentationBackend(ABC):
    name: str

    @abstractmethod
    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        """Return candidate masks in the original image coordinate space."""

    def health(self) -> dict[str, Any]:
        return {
            "available": True,
            "model_loaded": False,
            "message": None,
        }

