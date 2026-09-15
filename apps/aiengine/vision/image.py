"""Image transforms. Frames are ``uint8`` HWC RGB unless stated otherwise."""
from __future__ import annotations

import numpy as np

# ITU-R BT.601 luma coefficients — what CCTV encoders assume.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Accept grayscale, RGB or RGBA and always return 3-channel RGB."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    if frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2)
    return frame


def rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    """Luminance plane as float32 in [0, 255]."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 2:
        return frame
    return frame[:, :, :3] @ LUMA


def rgb_to_hsv(frame: np.ndarray) -> np.ndarray:
    """HSV with H in [0, 360), S and V in [0, 1].

    Fire and smoke detection lives in this space — flame pixels occupy a narrow,
    highly saturated hue band that is far more stable than raw RGB.
    """
    rgb = np.asarray(ensure_rgb(frame), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    r_max = mask & (maxc == r)
    g_max = mask & (maxc == g) & ~r_max
    b_max = mask & (maxc == b) & ~r_max & ~g_max
    with np.errstate(invalid="ignore", divide="ignore"):
        hue[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6
        hue[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2
        hue[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4
    hue *= 60.0

    saturation = np.zeros_like(maxc)
    np.divide(delta, maxc, out=saturation, where=maxc > 1e-6)
    return np.stack([hue, saturation, maxc], axis=-1)


def rgb_to_ycrcb(frame: np.ndarray) -> np.ndarray:
    """YCrCb (BT.601). The Cr/Cb planes give a strong, illumination-robust
    signal for both skin and flame."""
    rgb = np.asarray(ensure_rgb(frame), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return np.stack([y, cr, cb], axis=-1)


def resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resample with half-pixel centre alignment."""
    image = np.asarray(image, dtype=np.float32)
    single_channel = image.ndim == 2
    if single_channel:
        image = image[:, :, None]

    src_h, src_w = image.shape[:2]
    height, width = max(int(height), 1), max(int(width), 1)
    if (src_h, src_w) == (height, width):
        return image[:, :, 0] if single_channel else image

    row = (np.arange(height, dtype=np.float32) + 0.5) * (src_h / height) - 0.5
    col = (np.arange(width, dtype=np.float32) + 0.5) * (src_w / width) - 0.5
    row = np.clip(row, 0, src_h - 1)
    col = np.clip(col, 0, src_w - 1)

    r0 = np.floor(row).astype(np.int32)
    c0 = np.floor(col).astype(np.int32)
    r1 = np.minimum(r0 + 1, src_h - 1)
    c1 = np.minimum(c0 + 1, src_w - 1)
    dr = (row - r0)[:, None, None]
    dc = (col - c0)[None, :, None]

    top = image[r0][:, c0] * (1 - dc) + image[r0][:, c1] * dc
    bottom = image[r1][:, c0] * (1 - dc) + image[r1][:, c1] * dc
    out = top * (1 - dr) + bottom * dr
    return out[:, :, 0] if single_channel else out


def letterbox(image: np.ndarray, height: int, width: int, fill: float = 114.0):
    """Resize preserving aspect ratio and pad — stops people looking stretched.

    Returns ``(image, scale, pad_x, pad_y)`` so detections can be mapped back
    to the original frame coordinates.
    """
    image = np.asarray(image, dtype=np.float32)
    src_h, src_w = image.shape[:2]
    scale = min(height / src_h, width / src_w)
    new_h, new_w = max(int(round(src_h * scale)), 1), max(int(round(src_w * scale)), 1)
    resized = resize_bilinear(image, new_h, new_w)

    channels = () if resized.ndim == 2 else (resized.shape[2],)
    canvas = np.full((height, width) + channels, fill, dtype=np.float32)
    pad_y = (height - new_h) // 2
    pad_x = (width - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def normalise(image: np.ndarray, mean=None, std=None) -> np.ndarray:
    """Scale to [0, 1] then standardise per channel."""
    array = np.asarray(image, dtype=np.float32)
    if array.max(initial=0) > 1.5:
        array = array / 255.0
    if mean is None and std is None:
        return array
    mean = IMAGENET_MEAN if mean is None else np.asarray(mean, dtype=np.float32)
    std = IMAGENET_STD if std is None else np.asarray(std, dtype=np.float32)
    return (array - mean) / std


def to_chw(image: np.ndarray) -> np.ndarray:
    """HWC → CHW, adding the channel axis for grayscale input."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array[None, :, :]
    return np.transpose(array, (2, 0, 1))


def to_batch(image: np.ndarray) -> np.ndarray:
    """Single HWC image → a 1-element NCHW batch."""
    return to_chw(image)[None, ...]


def crop(image: np.ndarray, box, padding: float = 0.0) -> np.ndarray:
    """Crop ``(x1, y1, x2, y2)`` with optional relative padding, clipped to bounds."""
    image = np.asarray(image)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if padding:
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding
        x1, y1, x2, y2 = x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h
    x1 = max(int(round(x1)), 0)
    y1 = max(int(round(y1)), 0)
    x2 = min(int(round(x2)), w)
    y2 = min(int(round(y2)), h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1) + image.shape[2:], dtype=image.dtype)
    return image[y1:y2, x1:x2]


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(int(round(3 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Separable Gaussian blur — two 1-D passes instead of one 2-D convolution."""
    if sigma <= 0:
        return np.asarray(image, dtype=np.float32)
    array = np.asarray(image, dtype=np.float32)
    kernel = gaussian_kernel1d(sigma)
    radius = len(kernel) // 2

    def convolve_axis(data, axis):
        padded = np.pad(
            data,
            [(radius, radius) if i == axis else (0, 0) for i in range(data.ndim)],
            mode="edge",
        )
        out = np.zeros_like(data)
        for offset, weight in enumerate(kernel):
            slicer = [slice(None)] * data.ndim
            slicer[axis] = slice(offset, offset + data.shape[axis])
            out += weight * padded[tuple(slicer)]
        return out

    return convolve_axis(convolve_axis(array, 0), 1)


def box_blur(image: np.ndarray, radius: int = 4) -> np.ndarray:
    """Fast mean blur via an integral image — used for privacy face masking."""
    array = np.asarray(image, dtype=np.float32)
    if radius < 1:
        return array
    single = array.ndim == 2
    if single:
        array = array[:, :, None]
    h, w, c = array.shape
    integral = np.zeros((h + 1, w + 1, c), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)[:, None]
    y1 = np.clip(ys + radius + 1, 0, h)[:, None]
    x0 = np.clip(xs - radius, 0, w)[None, :]
    x1 = np.clip(xs + radius + 1, 0, w)[None, :]

    total = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )
    area = ((y1 - y0) * (x1 - x0)).astype(np.float64)[:, :, None]
    out = (total / np.maximum(area, 1)).astype(np.float32)
    return out[:, :, 0] if single else out


def pixelate(image: np.ndarray, blocks: int = 8) -> np.ndarray:
    """Mosaic an image region — the stronger privacy redaction option."""
    array = np.asarray(image, dtype=np.float32)
    h, w = array.shape[:2]
    small = resize_bilinear(array, max(h // max(blocks, 1), 1), max(w // max(blocks, 1), 1))
    return resize_bilinear(small, h, w)


def sobel_gradients(gray: np.ndarray):
    """Sobel ``(gx, gy)`` — the basis of our HOG features and edge energy."""
    gray = np.asarray(gray, dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            window = padded[i : i + gray.shape[0], j : j + gray.shape[1]]
            gx += kx[i, j] * window
            gy += ky[i, j] * window
    return gx, gy
