"""Campy AI's own computer-vision primitives.

Everything an ordinary CCTV frame has to go through before it reaches a neural
network — colour conversion, resizing, background modelling, optical flow,
blob extraction, geometry and multi-object tracking — implemented directly on
NumPy so the platform has no dependency on OpenCV at inference time.  (OpenCV,
when installed, is used only to *decode* RTSP streams.)
"""
from .geometry import (  # noqa: F401
    box_area,
    box_center,
    box_iou,
    clip_box,
    non_max_suppression,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    scale_box,
)
from .image import (  # noqa: F401
    letterbox,
    normalise,
    resize_bilinear,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_ycrcb,
    to_chw,
)
from .motion import BackgroundModel, MotionState, frame_difference, lucas_kanade_flow  # noqa: F401
from .tracking import Track, TrackedObject, TrackerConfig  # noqa: F401

__all__ = [
    "resize_bilinear", "rgb_to_gray", "rgb_to_hsv", "rgb_to_ycrcb", "normalise",
    "letterbox", "to_chw", "BackgroundModel", "MotionState", "frame_difference",
    "lucas_kanade_flow", "box_iou", "non_max_suppression", "point_in_polygon",
    "polygon_area", "polygon_centroid", "box_area", "box_center", "clip_box",
    "scale_box", "Track", "TrackedObject", "TrackerConfig",
]
