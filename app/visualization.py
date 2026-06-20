from PIL import Image, ImageDraw, ImageFont

from app.schemas import ToolObject

PALETTE = [
    (46, 134, 171),
    (242, 100, 48),
    (124, 179, 66),
    (142, 68, 173),
    (245, 166, 35),
    (38, 166, 154),
]


def overlay_objects(image: Image.Image, objects: list[ToolObject]) -> Image.Image:
    output = image.convert("RGBA")
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for index, obj in enumerate(objects):
        color = PALETTE[index % len(PALETTE)]
        fill = (*color, 42)
        stroke = (*color, 230)

        if len(obj.outline) >= 3:
            draw.polygon(obj.outline, fill=fill)
            draw.line(obj.outline + [obj.outline[0]], fill=stroke, width=3)
        else:
            draw.rectangle(obj.bbox_xyxy, outline=stroke, width=3)

        if obj.refinement_bbox_xyxy:
            _draw_dashed_rectangle(draw, obj.refinement_bbox_xyxy, stroke, width=2, dash=18)

        label = f"{obj.label} {obj.score:.2f}"
        x, y = obj.bbox_xyxy[0], max(0, obj.bbox_xyxy[1] - 16)
        text_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(text_box, fill=(*color, 230))
        draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(output, overlay).convert("RGB")


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
) -> None:
    x1, y1, x2, y2 = box
    _draw_dashed_line(draw, (x1, y1), (x2, y1), fill, width, dash)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), fill, width, dash)
    _draw_dashed_line(draw, (x2, y2), (x1, y2), fill, width, dash)
    _draw_dashed_line(draw, (x1, y2), (x1, y1), fill, width, dash)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    position = 0
    while position < length:
        segment_end = min(position + dash, length)
        if (position // dash) % 2 == 0:
            draw.line(
                (
                    round(x1 + dx * position),
                    round(y1 + dy * position),
                    round(x1 + dx * segment_end),
                    round(y1 + dy * segment_end),
                ),
                fill=fill,
                width=width,
            )
        position += dash
