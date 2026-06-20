import argparse
import json
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.main import run_segmentation
from app.prompts import DEFAULT_TOOL_PROMPTS
from app.settings import BackendName, Settings
from app.visualization import overlay_objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment tools in a local image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--out", required=True, help="Path to write JSON output.")
    parser.add_argument(
        "--backend",
        choices=["mock", "opencv", "sam3_mlx", "roboflow_sam3"],
        default=None,
    )
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt list.")
    parser.add_argument(
        "--annotated",
        default=None,
        help="Optional path to write an annotated PNG.",
    )
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    active_settings = Settings()
    backend_name: BackendName = args.backend or active_settings.segmenter_backend
    prompts = _parse_cli_prompts(args.prompts)

    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    try:
        result = run_segmentation(
            image,
            prompts,
            min_score=args.min_score,
            return_masks=False,
            return_contours=True,
            max_image_side=active_settings.max_image_side,
            backend_name=backend_name,
            active_settings=active_settings,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        raise SystemExit(json.dumps(detail, indent=2)) from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")

    if args.annotated:
        annotated_path = Path(args.annotated)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_objects(image, result.objects).save(annotated_path)


def _parse_cli_prompts(raw_prompts: str | None) -> list[str]:
    if raw_prompts is None or not raw_prompts.strip():
        return DEFAULT_TOOL_PROMPTS
    return [prompt.strip() for prompt in raw_prompts.split(",") if prompt.strip()]


if __name__ == "__main__":
    main()
