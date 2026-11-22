"""Selection masks: rasterization, feathering, components, placement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def rasterize_selection(width: int, height: int, shapes: list[dict]) -> np.ndarray:
    """Rasterize additive/subtractive shapes into a 0/1 mask.

    Shape dict: {"type": "polygon"|"rectangle"|"ellipse", "points": [[x,y],...],
                  "op": "add"|"subtract"}
    Coordinates are pixel-center coordinates; points may lie outside the canvas.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    for shape in shapes:
        kind = shape.get("type", "polygon")
        pts = np.asarray(shape.get("points", []), dtype=np.float64)
        if len(pts) == 0:
            continue
        if kind == "polygon":
            region = _point_in_polygon(xx, yy, pts)
        elif kind == "rectangle":
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            region = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        elif kind == "ellipse":
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            rx, ry = max((x1 - x0) / 2.0, 1e-9), max((y1 - y0) / 2.0, 1e-9)
            region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        else:
            raise ValueError(f"unknown shape type: {kind}")
        if shape.get("op", "add") == "subtract":
            mask[region] = 0
        else:
            mask[region] = 1
    return mask


def _point_in_polygon(xx: np.ndarray, yy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    inside = np.zeros(xx.shape, dtype=bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        hit = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= hit
        j = i
    return inside


def soft_mask_from_hard(hard: np.ndarray, feather_px: float) -> np.ndarray:
    """Distance-based feathering inside (and slightly outside) the selection.

    Returns float mask in [0,1]: 1 deep inside, tapering toward 0 at the
    boundary over ``feather_px`` pixels. Holes are feathered from their rims.
    """
    hard = hard.astype(bool)
    if feather_px <= 0:
        return hard.astype(np.float64)
    dist_in = ndimage.distance_transform_edt(hard)
    dist_out = ndimage.distance_transform_edt(~hard)
    signed = dist_in - dist_out  # >0 inside, <0 outside
    soft = np.clip(0.5 + signed / (2.0 * float(feather_px)), 0.0, 1.0)
    return soft


def connected_components(hard: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """4-connected components (matches the 4-neighbour discretization)."""
    labels, n = ndimage.label(hard, structure=np.array([[0, 1, 0],
                                                        [1, 1, 1],
                                                        [0, 1, 0]]))
    info = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        info.append(
            {
                "label": int(lab),
                "pixels": int(ys.size),
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1],
                "touches_border": bool(
                    xs.min() == 0 or ys.min() == 0
                    or xs.max() == hard.shape[1] - 1
                    or ys.max() == hard.shape[0] - 1
                ),
            }
        )
    return labels, info


def holes(hard: np.ndarray) -> np.ndarray:
    """Binary mask of interior holes (0-regions not connected to the border)."""
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    filled = ndimage.binary_fill_holes(hard, structure=structure)
    return filled & (~hard.astype(bool))


@dataclass
class Placement:
    """Mapping from source pixels to target canvas coordinates."""
    offset_y: int
    offset_x: int

    def map(self, ys: np.ndarray, xs: np.ndarray):
        return ys + self.offset_y, xs + self.offset_x

    def invert(self, ty: np.ndarray, tx: np.ndarray):
        return ty - self.offset_y, tx - self.offset_x


def make_placement(hard_source: np.ndarray, anchor_xy: tuple[float, float],
                   target_xy: tuple[float, float]) -> Placement:
    """Offset placing source selection's anchor point onto target point."""
    ys, xs = np.where(hard_source)
    if ys.size == 0:
        raise ValueError("empty source selection")
    cx = xs.mean() if anchor_xy is None else float(anchor_xy[0])
    cy = ys.mean() if anchor_xy is None else float(anchor_xy[1])
    ox = int(round(float(target_xy[0]) - cx))
    oy = int(round(float(target_xy[1]) - cy))
    return Placement(offset_y=oy, offset_x=ox)


def place_mask(mask: np.ndarray, placement: Placement,
               target_shape: tuple[int, int]) -> np.ndarray:
    """Embed a source-resolution mask onto the target canvas (zero padded)."""
    h, w = target_shape[:2]
    out = np.zeros((h, w), dtype=mask.dtype)
    sh, sw = mask.shape[:2]
    y0, x0 = max(placement.offset_y, 0), max(placement.offset_x, 0)
    sy0, sx0 = max(-placement.offset_y, 0), max(-placement.offset_x, 0)
    hh = min(sh - sy0, h - y0)
    ww = min(sw - sx0, w - x0)
    if hh > 0 and ww > 0:
        out[y0:y0 + hh, x0:x0 + ww] = mask[sy0:sy0 + hh, sx0:sx0 + ww]
    return out
