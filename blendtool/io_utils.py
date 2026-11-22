"""Image IO helpers. All internal images are float64 RGB arrays in [0, 1]."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def save_rgb(path: str, arr: np.ndarray) -> None:
    arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="RGB").save(path)


def save_gray(path: str, arr: np.ndarray) -> None:
    arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="L").save(path)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def colorize_diverging(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Blue (negative) -> white (0) -> red (positive), input arbitrary float."""
    v = np.asarray(values, dtype=np.float64)
    if vmax is None:
        m = np.max(np.abs(v))
        vmax = m if m > 0 else 1.0
    t = np.clip(v / vmax, -1.0, 1.0)
    out = np.ones(v.shape + (3,), dtype=np.float64)
    pos = t >= 0
    neg = ~pos
    out[pos] = np.stack([np.ones_like(t[pos]), 1 - t[pos], 1 - t[pos]], axis=-1)
    out[neg] = np.stack([1 + t[neg], 1 + t[neg], np.ones_like(t[neg])], axis=-1)
    return out


def colorize_magnitude(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Black -> purple -> orange -> yellow heat map for non-negative fields."""
    v = np.asarray(values, dtype=np.float64)
    if vmax is None:
        vmax = v.max() if v.max() > 0 else 1.0
    t = np.clip(v / vmax, 0.0, 1.0)
    stops = np.array(
        [
            [0.00, 0.02, 0.05, 0.10],
            [0.28, 0.10, 0.45, 0.70],
            [0.70, 0.25, 0.40, 0.65],
            [0.95, 0.55, 0.15, 0.30],
            [1.00, 0.98, 0.60, 0.25],
        ]
    )
    idx = np.clip(np.digitize(t, stops[1:-1, 0]), 0, len(stops) - 2)
    denom = np.maximum(stops[idx + 1, 0] - stops[idx, 0], 1e-12)
    frac = ((t - stops[idx, 0]) / denom)[..., None]
    lower = stops[idx, 1:]
    upper = stops[idx + 1, 1:]
    rgb = lower * (1 - frac) + upper * frac
    return rgb
