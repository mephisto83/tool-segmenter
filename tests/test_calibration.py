from PIL import Image, ImageDraw

from app.calibration import detect_light_square_board


def test_detect_light_square_board_maps_corners_to_mm() -> None:
    image = Image.new("RGB", (300, 300), (20, 25, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 250, 250), fill=(235, 235, 235))

    calibration = detect_light_square_board(image, 556)

    assert calibration is not None
    assert calibration.board_size_mm == 556
    assert calibration.point_to_mm(calibration.corners_px[0]) == (0.0, 0.0)
    assert calibration.point_to_mm(calibration.corners_px[2]) == (556.0, 556.0)

