from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageView:
    name: str
    image: Image.Image
    description: str


def generate_image_views(image: Image.Image) -> list[ImageView]:
    rgb = np.array(image.convert("RGB"))
    return [
        ImageView("original", image.convert("RGB"), "Unmodified RGB image."),
        ImageView(
            "clahe_luminance",
            _clahe_luminance(rgb),
            "CLAHE on Lab luminance to reveal dark boundaries and metal edges.",
        ),
        ImageView(
            "grayscale_clahe",
            _grayscale_clahe(rgb),
            "Contrast-enhanced grayscale repeated to RGB channels.",
        ),
        ImageView(
            "color_boost",
            _color_boost(rgb),
            "Saturation/value boosted view for colored handles.",
        ),
        ImageView(
            "background_residual",
            _background_residual(rgb),
            "High-contrast residual against a blurred local background estimate.",
        ),
    ]


def _clahe_luminance(rgb: np.ndarray) -> Image.Image:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge([enhanced_l, a_channel, b_channel])
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))


def _grayscale_clahe(rgb: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def _color_boost(rgb: np.ndarray) -> Image.Image:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(boosted)


def _background_residual(rgb: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
    residual = cv2.absdiff(gray, blurred)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    residual = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(residual.astype(np.uint8))
    return Image.fromarray(cv2.cvtColor(residual, cv2.COLOR_GRAY2RGB))

