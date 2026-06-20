from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def test_segment_tools_endpoint_with_mock_backend() -> None:
    client = TestClient(app)
    image_buffer = BytesIO()
    Image.new("RGB", (160, 120), color=(240, 240, 240)).save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = client.post(
        "/segment-tools",
        files={"image": ("drawer.png", image_buffer, "image/png")},
        data={"prompts": '["screwdriver","hand tool","drill bit"]'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image"] == {"width": 160, "height": 120}
    assert payload["backend"] == "mock"
    assert len(payload["objects"]) >= 2
    first_object = payload["objects"][0]
    assert {"label", "bbox_xyxy", "bbox_xywh", "centroid_xy", "outline"} <= first_object.keys()

