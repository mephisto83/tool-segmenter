import json
import time
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from app.backends import (
    BackendUnavailable,
    MockSegmentationBackend,
    OpenCvBackgroundRefinedBackend,
    OpenCvToolBackend,
    RoboflowSam3Backend,
    Sam3MlxBackend,
    Sam3MultiViewBackend,
    SegmentationBackend,
)
from app.postprocess import candidates_to_response_objects, dedupe_candidates, filter_candidates
from app.prompts import DEFAULT_TOOL_PROMPTS
from app.schemas import HealthResponse, ImageInfo, SegmentResponse, TimingMs
from app.settings import BackendName, Settings
from app.visualization import overlay_objects

settings = Settings()
app = FastAPI(title="Tool Segmenter", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_backend_cache: dict[str, SegmentationBackend] = {}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    backend = get_backend(settings.segmenter_backend)
    status = backend.health()
    return HealthResponse(
        ok=bool(status.get("available", False)),
        backend=backend.name,
        model_loaded=bool(status.get("model_loaded", False)),
        model_dir_exists=bool(status.get("model_dir_exists", True)),
        message=status.get("message"),
    )


@app.post("/segment-tools", response_model=SegmentResponse)
async def segment_tools(
    image: Annotated[UploadFile, File()],
    prompts: Annotated[str | None, Form()] = None,
    min_score: Annotated[float | None, Form()] = None,
    return_masks: Annotated[bool, Form()] = False,
    return_contours: Annotated[bool, Form()] = True,
    max_image_side: Annotated[int | None, Form()] = None,
) -> SegmentResponse:
    pil_image = await _read_upload_image(image)
    selected_prompts = parse_prompts(prompts)
    return run_segmentation(
        pil_image,
        selected_prompts,
        min_score=min_score,
        return_masks=return_masks,
        return_contours=return_contours,
        max_image_side=max_image_side,
        backend_name=settings.segmenter_backend,
        active_settings=settings,
    )


@app.post("/annotate-tools")
async def annotate_tools(
    image: Annotated[UploadFile, File()],
    prompts: Annotated[str | None, Form()] = None,
    min_score: Annotated[float | None, Form()] = None,
    max_image_side: Annotated[int | None, Form()] = None,
) -> Response:
    pil_image = await _read_upload_image(image)
    result = run_segmentation(
        pil_image,
        parse_prompts(prompts),
        min_score=min_score,
        return_masks=False,
        return_contours=True,
        max_image_side=max_image_side,
        backend_name=settings.segmenter_backend,
        active_settings=settings,
    )
    annotated = overlay_objects(pil_image, result.objects)
    buffer = BytesIO()
    annotated.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


def run_segmentation(
    image: Image.Image,
    prompts: list[str],
    *,
    min_score: float | None,
    return_masks: bool,
    return_contours: bool,
    max_image_side: int | None,
    backend_name: BackendName,
    active_settings: Settings,
) -> SegmentResponse:
    del max_image_side
    started = time.perf_counter()
    backend = get_backend(backend_name, active_settings)

    try:
        model_started = time.perf_counter()
        candidates = backend.segment(image, prompts)
        model_ms = _elapsed_ms(model_started)
    except BackendUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": str(exc), "backend": backend.name},
        ) from exc

    postprocess_started = time.perf_counter()
    filtered = filter_candidates(
        candidates,
        min_score=active_settings.min_score if min_score is None else min_score,
        min_area_px=active_settings.min_area_px,
    )
    deduped = dedupe_candidates(
        filtered,
        mask_iou_threshold=active_settings.dedup_mask_iou_threshold,
        box_iou_threshold=active_settings.dedup_box_iou_threshold,
    )
    objects = candidates_to_response_objects(
        deduped,
        image_size=image.size,
        epsilon_ratio=active_settings.contour_epsilon_ratio,
        return_masks=return_masks,
        return_contours=return_contours,
    )
    postprocess_ms = _elapsed_ms(postprocess_started)

    return SegmentResponse(
        image=ImageInfo(width=image.size[0], height=image.size[1]),
        backend=backend.name,
        objects=objects,
        timing_ms=TimingMs(
            total=_elapsed_ms(started),
            model=model_ms,
            postprocess=postprocess_ms,
        ),
    )


def get_backend(
    backend_name: BackendName,
    active_settings: Settings | None = None,
) -> SegmentationBackend:
    active_settings = active_settings or settings
    if backend_name in _backend_cache:
        return _backend_cache[backend_name]

    if backend_name == "mock":
        backend: SegmentationBackend = MockSegmentationBackend()
    elif backend_name == "opencv":
        backend = OpenCvToolBackend()
    elif backend_name == "opencv_bg_refined":
        backend = OpenCvBackgroundRefinedBackend()
    elif backend_name == "sam3_mlx":
        backend = Sam3MlxBackend(active_settings.model_dir)
    elif backend_name == "sam3_multiview":
        backend = Sam3MultiViewBackend(active_settings)
    elif backend_name == "roboflow_sam3":
        backend = RoboflowSam3Backend(
            active_settings.roboflow_base_url,
            active_settings.roboflow_api_key,
            active_settings.roboflow_api_key_file,
        )
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    _backend_cache[backend_name] = backend
    return backend


def parse_prompts(raw_prompts: str | None) -> list[str]:
    if raw_prompts is None or not raw_prompts.strip():
        return DEFAULT_TOOL_PROMPTS
    try:
        parsed = json.loads(raw_prompts)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw_prompts.split(",")]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=422, detail="prompts must be a JSON array of strings")
    prompts = [item.strip() for item in parsed if item.strip()]
    return prompts or DEFAULT_TOOL_PROMPTS


async def _read_upload_image(upload: UploadFile) -> Image.Image:
    try:
        data = await upload.read()
        return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="image must be a readable RGB image") from exc


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
