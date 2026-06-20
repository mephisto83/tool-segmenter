import numpy as np

from app.backends.base import SegmentationCandidate
from app.postprocess import dedupe_candidates, mask_to_bbox


def test_dedupe_removes_lower_scoring_overlap() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True
    candidates = [
        _candidate("screwdriver", "screwdriver", 0.90, mask),
        _candidate("hand tool", "hand tool", 0.60, mask.copy()),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].label == "screwdriver"


def test_dedupe_prefers_specific_label_when_scores_are_close() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True
    candidates = [
        _candidate("hand tool", "hand tool", 0.90, mask),
        _candidate("blue handle screwdriver", "blue handle screwdriver", 0.87, mask.copy()),
    ]
    deduped = dedupe_candidates(candidates, score_close_delta=0.05)
    assert len(deduped) == 1
    assert deduped[0].label == "blue handle screwdriver"


def _candidate(label: str, prompt: str, score: float, mask: np.ndarray) -> SegmentationCandidate:
    bbox = mask_to_bbox(mask)
    assert bbox is not None
    return SegmentationCandidate(label, prompt, score, bbox, mask)

