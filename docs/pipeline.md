# Pipeline Guide

This guide explains how to run and reason about the available segmentation pipelines.

## 1. OpenCV Baseline

Command:

```bash
python -m app.cli.segment_image \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out sample_outputs/opencv_sample.json \
  --backend opencv \
  --annotated sample_outputs/opencv_sample.png
```

Use this when:

- no API key is available
- you want fast local output
- you want proposal boxes for SAM-style refinement

Expected quality:

- good for colored handles and long metal bits
- partial for black-on-black objects
- weaker semantic labels

## 2. Roboflow SAM3

Command:

```bash
ROBOFLOW_API_KEY_FILE=~/Documents/roboflow/apikey \
python -m app.cli.segment_image \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out sample_outputs/roboflow_sam3_sample.json \
  --backend roboflow_sam3 \
  --prompts "screwdriver,tool bit,scissors,pliers,hand tool" \
  --annotated sample_outputs/roboflow_sam3_sample.png
```

Use this when:

- you want real SAM3 masks now
- you have a Roboflow API key
- hosted inference is acceptable

Expected quality:

- much better full-object outlines
- better black-object handling
- prompt-sensitive object count
- may still require prompt tuning and dedupe thresholds

## 3. OpenCV Background-Refined Experiment

Command:

```bash
python -m app.cli.segment_image \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out sample_outputs/opencv_bg_refined_sample.json \
  --backend opencv_bg_refined \
  --annotated sample_outputs/opencv_bg_refined_sample.png
```

Use this when:

- you want to compare background inversion against conservative OpenCV
- you are tuning classical CV behavior

Expected quality:

- sometimes recovers more of partial tools
- sometimes bridges adjacent tools
- not the recommended default

## 4. Multiview Crop Export

Command:

```bash
python -m app.cli.export_multiview_regions \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out-dir sample_outputs/multiview_pack \
  --backend opencv
```

This writes:

- `views/original.png`
- `views/clahe_luminance.png`
- `views/grayscale_clahe.png`
- `views/color_boost.png`
- `views/background_residual.png`
- `crops/<object-id>/<view-name>.png`
- `manifest.json`

Use this when:

- you want to inspect what SAM3 should receive
- you are implementing local SAM3 crop refinement
- you are comparing image preprocessing strategies

## 5. Calibrated White-Board Measurement

Command:

```bash
ROBOFLOW_API_KEY_FILE=~/Documents/roboflow/apikey \
python -m app.cli.segment_calibrated_image \
  --image /path/to/white_board_tools.jpeg \
  --out sample_outputs/calibrated_sam3.json \
  --backend roboflow_sam3 \
  --board-size-mm 556 \
  --prompts "screwdriver,tool bit,scissors,pliers,hand tool" \
  --annotated sample_outputs/calibrated_sam3.png
```

Use this when:

- the tools are on a known square white board
- approximate real-world positions and sizes are useful
- you want to filter SAM3 detections to the board instead of a dark drawer mat

How it works:

1. Detect the largest light square in the image.
2. Fit an ordered four-corner board polygon.
3. Compute a homography from image pixels to board millimeters.
4. Run SAM3 with board filtering.
5. Add per-object millimeter boxes and centroids to the JSON.

The general Roboflow CLI uses `ROBOFLOW_FILTER_MODE=auto` by default. Auto mode chooses light-board filtering when it detects a large white board and otherwise uses dark drawer-mat filtering. The calibrated CLI still requests board filtering directly because millimeter output depends on that board being present.

The measurement coordinate system is `board_mm_top_left_origin`: `(0, 0)` is the detected top-left board corner, and `(556, 556)` is the bottom-right board corner for a 556 mm board.

Limitations:

- Measurements are approximate and depend on accurate board detection.
- Tall objects, shadows, backlighting, and perspective can affect masks.
- `bbox_mm_xywh` is based on the axis-aligned transformed bbox corners, not a physical caliper measurement of the object.

## Prompt Strategy

Good starting prompts:

```text
screwdriver,tool bit,scissors,pliers,hand tool
```

More specific prompts can improve labels but also increase duplicates:

```text
phillips screwdriver,flathead screwdriver,precision screwdriver,drill bit,bit holder
```

General prompts such as `hand tool` improve recall, but they can produce generic duplicates. Dedupe handles many overlaps, but prompt choice still matters.

## Reading The JSON

For each object:

- `bbox_xyxy`: tight box around the returned mask
- `bbox_xywh`: same box as x/y/width/height
- `bbox_normalized_xyxy`: normalized tight box
- `refinement_bbox_xyxy`: padded box intended for SAM-like refinement
- `centroid_xy`: mask centroid
- `area_px`: mask area
- `outline`: external polygon
- `holes`: internal contours
- `mask_rle`: optional run-length encoded mask

For `roboflow_sam3`, masks come from SAM3 polygons.

For `opencv`, masks come from local color/brightness/geometry proposals.

## Common Failure Modes

### Missing Roboflow key

Use:

```bash
export ROBOFLOW_API_KEY_FILE=~/Documents/roboflow/apikey
```

or:

```bash
export ROBOFLOW_API_KEY=...
```

### Too many duplicate detections

Raise:

- `DEDUP_MASK_IOU_THRESHOLD`
- `DEDUP_BOX_IOU_THRESHOLD`

or reduce broad prompts.

### Missing small bits

Lower:

- `MIN_SCORE`
- `MIN_AREA_PX`

Add prompts:

- `tool bit`
- `drill bit`
- `screwdriver bit`

### Off-surface detections

The Roboflow backend defaults to automatic work-surface filtering. It keeps detections on a detected light board when present; otherwise it filters against the detected drawer mat. If drawer filtering is too strict or too loose, tune `min_overlap_ratio` in `_filter_to_drawer_mat()`. For unusual setups, set `ROBOFLOW_FILTER_MODE=none` to inspect raw SAM3 detections.

## Recommended Development Path

1. Run `opencv` locally to validate install.
2. Run `roboflow_sam3` to get real SAM3 masks.
3. Compare annotated PNGs.
4. Export multiview regions.
5. If implementing local SAM3, wire the local model into `Sam3MultiViewBackend.segment()`.
6. Keep the public response schema stable.
