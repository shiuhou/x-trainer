"""Robot-only image references; never infer or access task object state."""

import numpy as np
from PIL import Image, ImageDraw


def project_reference(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project ROS-camera (+Z forward, +X right, +Y down) metric points."""
    if (
        points_camera.shape != (3, 3)
        or intrinsics.shape != (3, 3)
        or not np.isfinite(points_camera).all()
        or not np.isfinite(intrinsics).all()
        or np.any(points_camera[:, 2] <= 0)
    ):
        raise ValueError("invalid_visual_reference_projection")
    homogeneous = points_camera @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def draw_reference(rgb: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Annotate a copy, preserving raw camera evidence and tensor contract."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("invalid_visual_reference_image")
    if pixels.shape != (3, 2) or not np.isfinite(pixels).all():
        raise ValueError("invalid_visual_reference_pixels")
    height, width = rgb.shape[:2]
    if np.any(pixels < 0) or np.any(pixels >= [width, height]):
        raise ValueError("visual_reference_outside_camera")
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    center = tuple(pixels[0])
    x, y = center
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="yellow", width=2)
    for endpoint, label, color in zip(
        pixels[1:], ("+X 20mm", "+Y 20mm"), ("lime", "cyan")
    ):
        end = tuple(endpoint)
        draw.line((center, end), fill=color, width=2)
        draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=color)
        draw.text(
            (end[0] + 5, end[1] + 5),
            label,
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    draw.text(
        (12, 12),
        "GRIP PLANE (robot reference, not object target)",
        fill="yellow",
        stroke_width=1,
        stroke_fill="black",
    )
    return np.asarray(image).copy()
