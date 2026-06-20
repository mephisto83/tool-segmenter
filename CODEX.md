# Codex Handoff

This repo is a FastAPI tool-drawer segmentation service. It can run locally with a classical OpenCV backend, and it can run real SAM3 segmentation through Roboflow's hosted SAM3 endpoint when a Roboflow API key is provided.

## What To Do First

```bash
git clone https://github.com/mephisto83/tool-segmenter.git
cd tool-segmenter

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

If `python3.12` is unavailable, Python 3.11 also works for the FastAPI/OpenCV/Roboflow path.

## Smoke Test Without Any API Key

Run OpenCV segmentation on the checked-in sample image:

```bash
python -m app.cli.segment_image \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out sample_outputs/opencv_sample.json \
  --backend opencv \
  --annotated sample_outputs/opencv_sample.png
```

This validates the app and produces an annotated image, but it is not as good as SAM3.

## Get A Roboflow API Key

1. Create or sign into a Roboflow account.
2. Open Roboflow account settings/API settings.
3. Copy the API key.
4. Save it outside the repo, for example:

```bash
mkdir -p ~/Documents/roboflow
printf '%s' 'PASTE_KEY_HERE' > ~/Documents/roboflow/apikey
chmod 600 ~/Documents/roboflow/apikey
```

Do not commit the key. Use either `ROBOFLOW_API_KEY_FILE` or `ROBOFLOW_API_KEY`.

## Run Real SAM3 Through Roboflow

```bash
ROBOFLOW_API_KEY_FILE=~/Documents/roboflow/apikey \
python -m app.cli.segment_image \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out sample_outputs/roboflow_sam3_sample.json \
  --backend roboflow_sam3 \
  --prompts "screwdriver,tool bit,scissors,pliers,hand tool" \
  --annotated sample_outputs/roboflow_sam3_sample.png
```

Expected behavior:

- The backend calls `https://serverless.roboflow.com/sam3/concept_segment`.
- It receives polygon masks from SAM3.
- It filters detections to the dark drawer mat so off-drawer objects are ignored.
- It writes JSON plus an annotated PNG.

## Run The API Server

OpenCV:

```bash
SEGMENTER_BACKEND=opencv uvicorn app.main:app --reload
```

Roboflow SAM3:

```bash
SEGMENTER_BACKEND=roboflow_sam3 \
ROBOFLOW_API_KEY_FILE=~/Documents/roboflow/apikey \
uvicorn app.main:app --reload
```

Then test:

```bash
curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/segment-tools" \
  -F "image=@sample_inputs/tool_drawer_sample.jpeg" \
  -F 'prompts=["screwdriver","tool bit","scissors","pliers","hand tool"]'
```

## Local MLX SAM3 Status

A Python 3.13 MLX SAM3 package can be installed separately, but local model execution is not the default path yet. The currently working SAM3 path is `roboflow_sam3`.

The local MLX path needs:

- Apple Silicon Mac
- Python 3.13+
- MLX SAM3 package
- SAM3 image model weights
- Implementation of the local crop/box/text API inside `app/backends/sam3_multiview_backend.py`

## Useful Commands

```bash
pytest
ruff check .

python -m app.cli.export_multiview_regions \
  --image sample_inputs/tool_drawer_sample.jpeg \
  --out-dir sample_outputs/multiview_pack \
  --backend opencv
```

## Important Files

- `app/backends/roboflow_sam3_backend.py`: working hosted SAM3 adapter.
- `app/backends/opencv_backend.py`: local proposal generation and drawer-mat filtering helpers.
- `app/backends/sam3_multiview_backend.py`: future local SAM3 multiview orchestration.
- `sample_inputs/tool_drawer_sample.jpeg`: metadata-stripped sample image.
- `.env.example`: environment variable reference.
- `docs/architecture.md`: decisions, backend boundaries, and data flow.
- `docs/pipeline.md`: how to run and compare each segmentation pipeline.
