from pydantic import BaseModel, Field, field_validator


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class TimingMs(BaseModel):
    total: int = Field(ge=0)
    model: int = Field(ge=0)
    postprocess: int = Field(ge=0)


class ToolObject(BaseModel):
    id: str
    label: str
    source_prompt: str
    score: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[int, int, int, int]
    bbox_xywh: tuple[int, int, int, int]
    bbox_normalized_xyxy: tuple[float, float, float, float]
    centroid_xy: tuple[int, int]
    area_px: int = Field(ge=0)
    outline: list[tuple[int, int]]
    holes: list[list[tuple[int, int]]] = Field(default_factory=list)
    mask_rle: list[int] | None = None

    @field_validator("bbox_xyxy", "bbox_xywh")
    @classmethod
    def validate_box(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if any(coord < 0 for coord in value):
            raise ValueError("box coordinates must be non-negative")
        return value

    @field_validator("bbox_normalized_xyxy")
    @classmethod
    def validate_normalized_box(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if any(coord < 0.0 or coord > 1.0 for coord in value):
            raise ValueError("normalized box coordinates must be between 0 and 1")
        return value


class SegmentResponse(BaseModel):
    image: ImageInfo
    backend: str
    objects: list[ToolObject]
    timing_ms: TimingMs


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    model_loaded: bool
    model_dir_exists: bool
    message: str | None = None

