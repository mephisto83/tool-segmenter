from pathlib import Path
from typing import Any

from PIL import Image

from app.backends.base import BackendUnavailable, SegmentationBackend, SegmentationCandidate


class Sam3MlxBackend(SegmentationBackend):
    name = "sam3_mlx"

    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)
        self._model: Any | None = None
        self._load_error: str | None = None

    def segment(self, image: Image.Image, prompts: list[str]) -> list[SegmentationCandidate]:
        del image, prompts
        self._ensure_loaded()
        raise BackendUnavailable(
            "SAM3/MLX model loading succeeded, but the local package API has not been mapped yet. "
            "Inspect the installed SAM3/MLX package and implement Sam3MlxBackend.segment()."
        )

    def health(self) -> dict[str, Any]:
        model_dir_exists = self.model_dir.exists()
        return {
            "available": self._model is not None,
            "model_loaded": self._model is not None,
            "model_dir_exists": model_dir_exists,
            "message": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise BackendUnavailable(self._load_error)
        if not self.model_dir.exists():
            self._load_error = (
                f"MODEL_DIR does not exist: {self.model_dir}. Download the SAM3/MLX image model "
                "there or run with SEGMENTER_BACKEND=mock."
            )
            raise BackendUnavailable(self._load_error)
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional local MLX install
            self._load_error = (
                "MLX is not importable. Install optional local dependencies with "
                "`pip install -e '.[sam3-mlx]'`, then inspect the SAM3 package API."
            )
            raise BackendUnavailable(self._load_error) from exc

        self._load_error = (
            "SAM3/MLX adapter is intentionally not hard-coded yet. The model directory exists and "
            "MLX imports, but the installed SAM3 image API must be inspected before wiring calls."
        )
        raise BackendUnavailable(self._load_error)

