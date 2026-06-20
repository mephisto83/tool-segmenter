import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from app.backends.sam3_multiview_backend import crop_views_for_candidate
from app.image_views import generate_image_views
from app.main import run_segmentation
from app.prompts import DEFAULT_TOOL_PROMPTS
from app.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export OpenCV proposal regions and manipulated image views for SAM3 refinement."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", required=True, help="Directory to write the region pack.")
    parser.add_argument("--backend", default="opencv", choices=["opencv", "opencv_bg_refined"])
    args = parser.parse_args()

    settings = Settings()
    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    out_dir = Path(args.out_dir)
    views_dir = out_dir / "views"
    crops_dir = out_dir / "crops"
    views_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    views = generate_image_views(image)
    for view in views:
        view.image.save(views_dir / f"{view.name}.png")

    proposals = run_segmentation(
        image,
        DEFAULT_TOOL_PROMPTS,
        min_score=None,
        return_masks=False,
        return_contours=True,
        max_image_side=settings.max_image_side,
        backend_name=args.backend,
        active_settings=settings,
    )

    manifest: dict[str, object] = {
        "image": proposals.image.model_dump(mode="json"),
        "backend": args.backend,
        "views": [
            {"name": view.name, "description": view.description, "path": f"views/{view.name}.png"}
            for view in views
        ],
        "regions": [],
    }

    regions: list[dict[str, object]] = []
    for obj in proposals.objects:
        bbox = obj.refinement_bbox_xyxy or obj.bbox_xyxy
        object_dir = crops_dir / obj.id
        object_dir.mkdir(parents=True, exist_ok=True)
        crop_paths: dict[str, str] = {}
        for name, crop in crop_views_for_candidate(views, bbox).items():
            crop_path = object_dir / f"{name}.png"
            crop.save(crop_path)
            crop_paths[name] = str(crop_path.relative_to(out_dir))

        regions.append(
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "bbox_xyxy": obj.bbox_xyxy,
                "refinement_bbox_xyxy": bbox,
                "crop_paths": crop_paths,
            }
        )

    manifest["regions"] = regions
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
