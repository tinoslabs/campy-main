"""Turn a stored frame and a detection box into a picture that points at it.

A report that says "unauthorised entry, 21:14, Assembly Line A" is a claim. The
same line beside the frame, with the person outlined and the zone they crossed
drawn on it, is something a manager can check in a second and an investigator
can put in front of somebody.
"""
from __future__ import annotations

import io
import logging

logger = logging.getLogger("campy.evidence")

SEVERITY_COLOURS = {
    "critical": (200, 40, 32),
    "high": (200, 40, 32),
    "medium": (196, 106, 12),
    "low": (26, 132, 120),
    "info": (26, 132, 120),
}
FALLBACK_COLOUR = (196, 106, 12)


def _box_in_image(box, image_width: int, image_height: int, frame_width: int, frame_height: int):
    """Scale a detection box onto the stored image, whatever size it was saved at."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    # Boxes are recorded in the analysed frame's pixels. The snapshot is
    # normally that same frame, but scale anyway rather than trust it.
    sx = image_width / frame_width if frame_width else 1.0
    sy = image_height / frame_height if frame_height else 1.0
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def annotate_event(event, *, max_width: int = 900) -> bytes | None:
    """Render an event's snapshot with the detection marked, as JPEG bytes.

    Returns None when there is no stored frame — evidence is best-effort, and a
    report is still worth producing without it.
    """
    snapshot = getattr(event, "snapshot", None)
    if snapshot is None or not getattr(snapshot, "image", None):
        return None

    try:
        from PIL import Image, ImageDraw
    except ImportError:                                   # pragma: no cover
        return None

    try:
        with snapshot.image.open("rb") as handle:
            image = Image.open(handle)
            image.load()
        image = image.convert("RGB")
    except Exception:                                     # noqa: BLE001 - evidence is optional
        logger.warning("Could not read the snapshot for event %s", event.pk)
        return None

    colour = SEVERITY_COLOURS.get(getattr(event, "severity", ""), FALLBACK_COLOUR)
    draw = ImageDraw.Draw(image)

    # The zone the event happened in, so the boundary that was crossed is visible.
    zone = getattr(event, "zone", None)
    if zone is not None and getattr(zone, "polygon", None):
        from apps.aiengine.vision.geometry import as_polygon

        points = as_polygon(zone.polygon)
        if len(points) >= 3:
            scaled = [(x * image.width, y * image.height) for x, y in points]
            draw.polygon(scaled, outline=(120, 180, 255))

    box = _box_in_image(
        getattr(event, "bounding_box", None) or [],
        image.width, image.height,
        getattr(snapshot, "width", 0) or image.width,
        getattr(snapshot, "height", 0) or image.height,
    )
    if box is not None:
        x1, y1, x2, y2 = box
        width = max(2, round(image.width / 240))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)

        from apps.analytics.pdf import humanise

        label = humanise(getattr(event, "event_type", ""))
        confidence = getattr(event, "confidence", None)
        if confidence:
            label = f"{label} · {float(confidence) * 100:.0f}%"
        _draw_label(draw, label, x1, y1, colour, image.width)

    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def _draw_label(draw, text: str, x: float, y: float, colour, image_width: int) -> None:
    """A filled caption above the box, nudged inside the frame if it would clip."""
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=max(11, round(image_width / 55)))
    except TypeError:                                     # Pillow < 10.1
        font = ImageFont.load_default()

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = 4
    width, height = right - left + pad * 2, bottom - top + pad * 2
    origin_y = y - height if y - height >= 0 else y
    origin_x = min(x, max(0, image_width - width))
    draw.rectangle([origin_x, origin_y, origin_x + width, origin_y + height], fill=colour)
    draw.text((origin_x + pad - left, origin_y + pad - top), text, fill=(255, 255, 255), font=font)
