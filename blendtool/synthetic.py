"""Synthetic images with analytically known gradients, for verification."""
from __future__ import annotations

import numpy as np


def planar_ramp(height: int, width: int, slope_y: float, slope_x: float,
                base: float = 0.5) -> np.ndarray:
    """f(y,x) = base + sy*y + sx*x; exact gradients are the slopes."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    f = base + slope_y * yy + slope_x * xx
    return np.repeat(f[..., None], 3, axis=2)


def sine_pattern(height: int, width: int, freq: float = 0.25,
                 amp: float = 0.2, base: float = 0.5) -> np.ndarray:
    """f = base + amp*sin(2*pi*freq*x); analytic x-gradient known."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    f = base + amp * np.sin(2 * np.pi * freq * xx)
    return np.repeat(f[..., None], 3, axis=2)


def checkerboard(height: int, width: int, cell: int = 8,
                 v0: float = 0.2, v1: float = 0.8) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    f = np.where(((xx // cell) + (yy // cell)) % 2 == 0, v0, v1)
    return np.repeat(f[..., None].astype(np.float64), 3, axis=2)


def flat_field(height: int, width: int, level: float) -> np.ndarray:
    return np.full((height, width, 3), level, dtype=np.float64)


def constant_offset(img: np.ndarray, offset: float) -> np.ndarray:
    """Same gradients as img, different absolute level."""
    return img + offset


def analytic_gradient_ramp(slope_y: float, slope_x: float,
                           edges: dict, width: int) -> np.ndarray:
    """Exact per-edge guidance vectors for planar_ramp (grayscale repeated)."""
    iy, ix = divmod(edges["i"], width)
    jy, jx = divmod(edges["j"], width)
    g_lum = slope_y * (jy - iy) + slope_x * (jx - ix)
    return np.repeat(g_lum[:, None].astype(np.float64), 3, axis=1)


def analytic_gradient_sine(freq: float, amp: float, edges: dict,
                           width: int) -> np.ndarray:
    """Exact per-edge guidance for sine_pattern (unit-step differences)."""
    iy, ix = divmod(edges["i"], width)
    jy, jx = divmod(edges["j"], width)
    f = lambda x: amp * np.sin(2 * np.pi * freq * x)
    g_lum = f(jx.astype(np.float64)) - f(ix.astype(np.float64))
    return np.repeat(g_lum[:, None], 3, axis=1)


def rect_mask(height: int, width: int, y0: int, x0: int, y1: int,
              x1: int) -> np.ndarray:
    m = np.zeros((height, width), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def mask_with_hole(height: int, width: int) -> np.ndarray:
    m = np.zeros((height, width), dtype=bool)
    m[8:40, 8:40] = True
    m[20:28, 20:28] = False
    return m


def two_components(height: int, width: int) -> np.ndarray:
    m = np.zeros((height, width), dtype=bool)
    m[8:20, 8:20] = True
    m[30:42, 30:42] = True
    return m


def full_canvas_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)
