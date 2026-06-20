from pathlib import Path
from typing import Any

from PIL import Image

from app.backends.base import BackendUnavailable, SegmentationBackend, SegmentationCandidate
from app.backends.opencv_backend import OpenCvToolBackend
from app.image_views import ImageView, generate_image_views
from app.settings import Settings


class Sam3MultiViewBackend(SegmentationBackend):
    name = "sam3_multiview"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proposal_backend = OpenCvToolBackend()
        self._load_error: str | None = None

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        proposals = self.proposal_backend.segment(image, prompts)
        views = generate_image_views(image)
        self._ensure_sam3_available()
        del proposals, views
        raise BackendUnavailable(
            "SAM3 multiview orchestration is ready, but the local SAM3 adapter has not been "
            "mapped to an installed package API yet."
        )

    def health(self) -> dict[str, Any]:
        model_dir = Path(self.settings.model_dir)
        sam_available = self._sam3_import_available() and model_dir.exists()
        message = None if sam_available else self._missing_sam3_message(model_dir)
        return {
            "available": sam_available,
            "model_loaded": False,
            "model_dir_exists": model_dir.exists(),
            "message": message,
        }

    def _ensure_sam3_available(self) -> None:
        model_dir = Path(self.settings.model_dir)
        if not model_dir.exists() or not self._sam3_import_available():
            self._load_error = self._missing_sam3_message(model_dir)
            raise BackendUnavailable(self._load_error)

    def _sam3_import_available(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
        except Exception:
            return False
        return True

    def _missing_sam3_message(self, model_dir: Path) -> str:
        if not model_dir.exists():
            return (
                f"MODEL_DIR does not exist: {model_dir}. The multiview pipeline can export "
                "OpenCV proposals and manipulated image crops, but SAM3 refinement needs a local "
                "SAM3/MLX model directory."
            )
        return (
            "MLX/SAM3 is not importable in this environment. The multiview proposal pipeline is "
            "implemented; install the SAM3/MLX package and map its region/crop API in "
            "Sam3MultiViewBackend.segment()."
        )


def crop_views_for_candidate(
    views: list[ImageView],
    bbox: tuple[int, int, int, int],
) -> dict[str, Image.Image]:
    return {view.name: view.image.crop(bbox) for view in views}

