from PIL import Image

from app.backends.sam3_multiview_backend import Sam3MultiViewBackend, crop_views_for_candidate
from app.image_views import generate_image_views
from app.settings import Settings


def test_generate_image_views_preserves_size_and_names() -> None:
    image = Image.new("RGB", (64, 48), color=(30, 80, 140))
    views = generate_image_views(image)

    assert [view.name for view in views] == [
        "original",
        "clahe_luminance",
        "grayscale_clahe",
        "color_boost",
        "background_residual",
    ]
    assert all(view.image.size == image.size for view in views)


def test_crop_views_for_candidate_returns_all_view_crops() -> None:
    image = Image.new("RGB", (64, 48), color=(30, 80, 140))
    views = generate_image_views(image)
    crops = crop_views_for_candidate(views, (10, 8, 30, 28))

    assert set(crops) == {view.name for view in views}
    assert all(crop.size == (20, 20) for crop in crops.values())


def test_sam3_multiview_health_reports_missing_local_model() -> None:
    settings = Settings(SEGMENTER_BACKEND="sam3_multiview", MODEL_DIR="/tmp/not-a-sam3-model")
    status = Sam3MultiViewBackend(settings).health()

    assert status["available"] is False
    assert status["model_dir_exists"] is False
    assert "SAM3" in status["message"]

