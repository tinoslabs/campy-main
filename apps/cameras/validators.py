"""Input validation shared by the zone form and the zone API."""
from __future__ import annotations
def clean_zone_polygon(points, error):
    """Validate and tidy a zone outline drawn over a camera view.

    Coordinates are stored relative to the frame (0-1) so a zone survives a
    change of stream resolution, and they must be numbers: a polygon carrying a
    string reaches every page and every worker that reads that zone. Points
    slightly outside the frame are clamped rather than refused — you cannot
    draw outside the image, so that is a rounding artefact, not an error.
    """
    if not isinstance(points, list) or len(points) < 3:
        raise error("Draw at least three points to define a zone.")

    cleaned = []
    for point in points:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            raise error("Each point must be an [x, y] pair.")
        pair = []
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise error("Zone coordinates must be numbers.")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise error("Zone coordinates must be numbers.")
            pair.append(min(max(value, 0.0), 1.0))
        cleaned.append(pair)
    return cleaned
