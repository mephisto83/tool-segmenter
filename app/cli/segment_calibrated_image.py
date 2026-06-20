import argparse
import json
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.calibration import detect_light_square_board
from app.main import run_segmentation
from app.prompts import DEFAULT_TOOL_PROMPTS
from app.settings import Settings
from app.visualization import overlay_objects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment a calibrated square-board image and write pixel plus mm results."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--annotated", default=None)
    parser.add_argument("--backend", default="roboflow_sam3")
    parser.add_argument("--prompts", default="screwdriver,tool bit,scissors,pliers,hand tool")
    parser.add_argument("--board-size-mm", type=float, default=556.0)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    settings = Settings(
        SEGMENTER_BACKEND=args.backend,
        ROBOFLOW_FILTER_MODE="light_board",
        CALIBRATION_BOARD_SIZE_MM=args.board_size_mm,
    )
    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    calibration = detect_light_square_board(image, args.board_size_mm)
    if calibration is None:
        raise SystemExit("Could not detect the light square calibration board.")

    try:
        result = run_segmentation(
            image,
            _parse_prompts(args.prompts),
            min_score=args.min_score,
            return_masks=False,
            return_contours=True,
            max_image_side=settings.max_image_side,
            backend_name=args.backend,
            active_settings=settings,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        raise SystemExit(json.dumps(detail, indent=2)) from exc

    payload = result.model_dump(mode="json")
    payload["calibration"] = {
        "board_size_mm": calibration.board_size_mm,
        "corners_px": calibration.corners_px,
        "homography_px_to_mm": calibration.homography_px_to_mm,
        "coordinate_space": "board_mm_top_left_origin",
    }
    for obj in payload["objects"]:
        bbox_mm = calibration.bbox_to_mm(tuple(obj["bbox_xyxy"]))
        centroid_mm = calibration.point_to_mm(tuple(obj["centroid_xy"]))
        obj["bbox_mm_xyxy"] = bbox_mm
        obj["bbox_mm_xywh"] = (
            round(bbox_mm[0], 2),
            round(bbox_mm[1], 2),
            round(bbox_mm[2] - bbox_mm[0], 2),
            round(bbox_mm[3] - bbox_mm[1], 2),
        )
        obj["centroid_mm"] = centroid_mm

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.annotated:
        annotated_path = Path(args.annotated)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_objects(image, result.objects).save(annotated_path)


def _parse_prompts(raw_prompts: str) -> list[str]:
    prompts = [prompt.strip() for prompt in raw_prompts.split(",") if prompt.strip()]
    return prompts or DEFAULT_TOOL_PROMPTS


if __name__ == "__main__":
    main()
