from PIL import Image, ImageDraw

from app.backends.opencv_backend import OpenCvToolBackend


def test_opencv_backend_detects_separate_colored_tools_on_dark_mat() -> None:
    image = Image.new("RGB", (320, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 35, 275, 385), fill=(18, 18, 18))
    draw.rectangle((80, 70, 130, 170), fill=(0, 145, 230))
    draw.rectangle((98, 170, 112, 330), fill=(185, 185, 185))
    draw.rectangle((190, 70, 240, 170), fill=(255, 90, 20))
    draw.rectangle((208, 170, 222, 330), fill=(185, 185, 185))

    candidates = OpenCvToolBackend().segment(image, [])

    assert len(candidates) >= 2
    assert all(candidate.mask.any() for candidate in candidates)

