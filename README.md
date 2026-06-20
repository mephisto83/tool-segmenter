# Tool Segmenter

Tool Segmenter is a backend-pluggable FastAPI service for segmenting hand tools and tool bits in a drawer photo. It returns one JSON object per detected item with a label, score, bounding boxes, centroid, pixel area, polygon outline, holes, and optional mask RLE.

The default backend is `opencv`, a lightweight pixel-based detector for drawer photos. The `mock` backend is still available so the API, CLI, postprocessing, visualization, and tests work before any model integration is installed.

## Setup

With `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

With `venv` and `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Copy the example environment file:

```bash
cp .env.example .env
```

## Run With OpenCV Backend

```bash
SEGMENTER_BACKEND=opencv uvicorn app.main:app --reload
```

The OpenCV backend finds the dark drawer mat, segments saturated colored handles plus bright metal shafts, groups aligned components into individual tool candidates, and returns polygon outlines. It also returns `refinement_bbox_xyxy`, a padded region clipped to the drawer mat for a SAM-style backend to refine when OpenCV only sees part of an object. It is useful for fast local debugging, but black tools on a black mat and semantic labels still need a promptable model backend such as SAM3.

There is also an experimental `opencv_bg_refined` backend. It estimates the drawer-mat background, inverts that background inside each padded refinement region, and runs crop-local GrabCut seeded by the OpenCV mask. This can recover more of partial objects, but it may bridge nearby tools in dense areas, so `opencv` remains the conservative default.

## SAM3 Multiview Refinement

The `sam3_multiview` backend is the intended high-quality refinement path:

1. Run conservative OpenCV to propose individual tool regions.
2. Expand each partial detection into `refinement_bbox_xyxy`.
3. Generate multiple views of the same image:
   - `original`
   - `clahe_luminance`
   - `grayscale_clahe`
   - `color_boost`
   - `background_residual`
4. Crop each view to the refinement box.
5. Ask SAM3 to segment the tool inside each crop.
6. Map masks back to original image coordinates, score them by consensus, and dedupe.

The repo does not hard-code the SAM3/MLX call yet because no local SAM3 package/model API is installed in this environment. Until that is available, `sam3_multiview` reports a clear unavailable message.

You can export the exact multiview crop pack that SAM3 should consume:

```bash
python -m app.cli.export_multiview_regions \
  --image /path/to/tool_drawer.jpg \
  --out-dir sample_outputs/multiview_pack \
  --backend opencv
```

The export writes full image views, per-object crop views, and `manifest.json` with each region's source bbox and padded refinement bbox.

## Run With Mock Backend

```bash
SEGMENTER_BACKEND=mock uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Segment an image:

```bash
curl -X POST "http://localhost:8000/segment-tools" \
  -F "image=@/path/to/tool_drawer.jpg" \
  -F 'prompts=["screwdriver","screwdriver bit","drill bit","scissors","pliers","hand tool"]' \
  -F "min_score=0.2"
```

Create an annotated PNG:

```bash
curl -X POST "http://localhost:8000/annotate-tools" \
  -F "image=@/path/to/tool_drawer.jpg" \
  --output sample_outputs/annotated.png
```

## CLI

```bash
python -m app.cli.segment_image \
  --image /path/to/tool_drawer.jpg \
  --out sample_outputs/output.json \
  --backend opencv \
  --prompts "screwdriver,drill bit,scissors" \
  --annotated sample_outputs/annotated.png \
  --min-score 0.25
```

## Local SAM3/MLX Backend

The `sam3_mlx` adapter is intentionally isolated and conservative. It does not guess at the local SAM3/MLX Python API. After downloading or installing the actual model/package, inspect the package entry points and implement `Sam3MlxBackend.segment()` around the real API.

Expected model location:

```bash
mkdir -p models/sam3-image
# Download the SAM3-compatible MLX image model files into ./models/sam3-image.
```

Optional local dependencies:

```bash
pip install -e '.[sam3-mlx]'
```

Run:

```bash
SEGMENTER_BACKEND=sam3_mlx MODEL_DIR=./models/sam3-image uvicorn app.main:app --reload
```

If the model directory or MLX dependency is missing, `/health` and `/segment-tools` return a clear unavailable message instead of a stack trace.

## Roboflow SAM3 Backend

The Roboflow adapter calls an isolated HTTP endpoint at `ROBOFLOW_BASE_URL`.

```bash
SEGMENTER_BACKEND=roboflow_sam3 \
ROBOFLOW_BASE_URL=http://localhost:9001 \
ROBOFLOW_API_KEY=... \
uvicorn app.main:app --reload
```

The adapter posts PNG bytes and prompts to `POST /segment` on that service, then maps predictions into the shared `SegmentationCandidate` interface.

## Example Response

```json
{
  "image": {
    "width": 1536,
    "height": 2048
  },
  "backend": "mock",
  "objects": [
    {
      "id": "obj_0001",
      "label": "blue handle screwdriver",
      "source_prompt": "blue handle screwdriver",
      "score": 0.88,
      "bbox_xyxy": [245, 286, 399, 1597],
      "bbox_xywh": [245, 286, 154, 1311],
      "bbox_normalized_xyxy": [0.1595, 0.1396, 0.2598, 0.7798],
      "refinement_bbox_xyxy": [210, 244, 434, 1639],
      "refinement_bbox_normalized_xyxy": [0.1367, 0.1191, 0.2826, 0.8003],
      "centroid_xy": [322, 941],
      "area_px": 201894,
      "outline": [[245, 286], [245, 1596], [398, 1596], [398, 286]],
      "holes": [],
      "mask_rle": null
    }
  ],
  "timing_ms": {
    "total": 12,
    "model": 1,
    "postprocess": 4
  }
}
```

## Prompt And Threshold Tuning

The default prompt set mixes generic and specific labels so multiple prompts can find the same object from different angles. Postprocessing deduplicates overlapping masks and prefers a more specific label, such as `blue handle screwdriver`, over `hand tool` when scores are close.

Useful environment knobs:

- `MIN_SCORE`
- `DEDUP_MASK_IOU_THRESHOLD`
- `DEDUP_BOX_IOU_THRESHOLD`
- `CONTOUR_EPSILON_RATIO`
- `MAX_IMAGE_SIDE`
- `MIN_AREA_PX`

## Tests And Lint

```bash
pytest
ruff check .
```
