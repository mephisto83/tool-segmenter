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

        label = f"{obj.label} {obj.score:.2f}"
        x, y = obj.bbox_xyxy[0], max(0, obj.bbox_xyxy[1] - 16)
        text_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(text_box, fill=(*color, 230))
        draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(output, overlay).convert("RGB")

