from app.schemas import ImageInfo, SegmentResponse, TimingMs, ToolObject


def test_segment_response_serializes_expected_shape() -> None:
    response = SegmentResponse(
        image=ImageInfo(width=100, height=200),
        backend="mock",
        objects=[
            ToolObject(
                id="obj_0001",
                label="screwdriver",
                source_prompt="screwdriver",
                score=0.91,
                bbox_xyxy=(10, 20, 30, 80),
                bbox_xywh=(10, 20, 20, 60),
                bbox_normalized_xyxy=(0.1, 0.1, 0.3, 0.4),
                centroid_xy=(20, 50),
                area_px=500,
                outline=[(10, 20), (30, 20), (30, 80), (10, 80)],
                holes=[],
                mask_rle=None,
            )
        ],
        timing_ms=TimingMs(total=12, model=8, postprocess=2),
    )

    payload = response.model_dump(mode="json")
    assert payload["image"] == {"width": 100, "height": 200}
    assert payload["objects"][0]["bbox_xyxy"] == [10, 20, 30, 80]
    assert payload["objects"][0]["mask_rle"] is None

