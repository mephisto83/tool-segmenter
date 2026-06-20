from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BackendName = Literal[
    "mock",
    "opencv",
    "opencv_bg_refined",
    "sam3_mlx",
    "sam3_multiview",
    "roboflow_sam3",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    segmenter_backend: BackendName = Field(default="opencv", validation_alias="SEGMENTER_BACKEND")
    model_dir: str = Field(default="./models/sam3-image", validation_alias="MODEL_DIR")
    roboflow_api_key: str = Field(default="", validation_alias="ROBOFLOW_API_KEY")
    roboflow_api_key_file: str = Field(default="", validation_alias="ROBOFLOW_API_KEY_FILE")
    roboflow_filter_mode: str = Field(default="drawer_mat", validation_alias="ROBOFLOW_FILTER_MODE")
    calibration_board_size_mm: float = Field(
        default=556.0,
        validation_alias="CALIBRATION_BOARD_SIZE_MM",
    )
    roboflow_base_url: str = Field(
        default="https://serverless.roboflow.com",
        validation_alias="ROBOFLOW_BASE_URL",
    )
    min_score: float = Field(default=0.20, validation_alias="MIN_SCORE")
    dedup_mask_iou_threshold: float = Field(
        default=0.70,
        validation_alias="DEDUP_MASK_IOU_THRESHOLD",
    )
    dedup_box_iou_threshold: float = Field(
        default=0.75,
        validation_alias="DEDUP_BOX_IOU_THRESHOLD",
    )
    contour_epsilon_ratio: float = Field(default=0.003, validation_alias="CONTOUR_EPSILON_RATIO")
    max_image_side: int = Field(default=1600, validation_alias="MAX_IMAGE_SIDE")
    min_area_px: int = Field(default=16, validation_alias="MIN_AREA_PX")
