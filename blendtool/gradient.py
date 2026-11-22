"""Guidance gradient fields and neighbour-edge construction.

All gradients use the standard 4-neighbour finite difference
``g(i->j) = v[j] - v[i]`` on a unit grid.
"""
from __future__ import annotations

import numpy as np


def build_edges(height: int, width: int, support: np.ndarray):
    """All right/down edges with at least one endpoint inside the support.

    Returns dict of int64 flat-index arrays (i -> j, j is right or below i):
      i, j            endpoint flat indices
      horizontal      bool, True for right edge, False for down edge
    """
    sup = support.ravel().astype(bool)
    right_i = np.arange(height * width, dtype=np.int64).reshape(height, width)[:, :-1]
    right_j = right_i + 1
    down_i = np.arange(height * width, dtype=np.int64).reshape(height, width)[:-1, :]
    down_j = down_i + width
    i = np.concatenate([right_i.ravel(), down_i.ravel()])
    j = np.concatenate([right_j.ravel(), down_j.ravel()])
    horizontal = np.concatenate(
        [np.ones(right_i.size, dtype=bool), np.zeros(down_i.size, dtype=bool)]
    )
    keep = sup[i] | sup[j]
    return {"i": i[keep], "j": j[keep], "horizontal": horizontal[keep]}


def guidance_field(source_embedded: np.ndarray,
                   target: np.ndarray,
                   edges: dict,
                   source_valid_edge: np.ndarray,
                   mask_i: np.ndarray,
                   mask_j: np.ndarray,
                   strategy: str,
                   soft: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge guidance vectors and per-edge scalar weights.

    Parameters
    ----------
    source_embedded : HxWx3 float, source image placed on the target canvas
                      (values only meaningful where source is inside bounds)
    target           : HxWx3 float target image
    source_valid_edge: bool[E], both endpoints map inside the source image
    mask_i, mask_j   : soft mask value (0..1) at each endpoint
    strategy         : "source" | "mixed" | "direct"
    soft             : whether the mask is a feathered (soft) selection

    Returns
    -------
    g       : (E, 3) guidance gradient for each edge
    w_edge  : (E,) least-squares row weights for gradient equations
    """
    i, j = edges["i"], edges["j"]
    src = source_embedded.reshape(-1, 3)
    tgt = target.reshape(-1, 3)
    ds = src[j] - src[i]
    dt = tgt[j] - tgt[i]

    valid = source_valid_edge[:, None]
    if strategy == "source":
        g = np.where(valid, ds, dt)
    elif strategy == "direct":
        g = dt
    elif strategy == "mixed":
        # Perez mixed gradients: keep the stronger of source/target edge.
        mag_s = np.linalg.norm(ds, axis=1)
        mag_t = np.linalg.norm(dt, axis=1)
        pick_s = (mag_s >= mag_t) & source_valid_edge
        g = np.where(pick_s[:, None], ds, dt)
        # Edges where source is out of bounds cannot use source gradients.
        g = np.where(valid, g, dt)
    else:
        raise ValueError(f"unknown guidance strategy: {strategy}")

    if soft:
        # Across the feather band the guidance eases from the chosen field
        # toward the target's own gradient, and equation weight tapers so the
        # solution relaxes back to target colors outside the support.
        w_endpoint = np.maximum(mask_i, mask_j)
        w_edge = w_endpoint
        g = w_endpoint[:, None] * g + (1.0 - w_endpoint[:, None]) * dt
    else:
        # Hard selection: full-weight Poisson equations everywhere; boundary
        # edges are anchored by Dirichlet target values in the solver.
        w_edge = np.ones(i.shape, dtype=np.float64)
    return g, w_edge


def edge_source_validity(edges: dict, source_bounds: tuple[int, int, int, int],
                         height: int, width: int) -> np.ndarray:
    """True for edges whose both endpoints lie inside the placed source rect.

    source_bounds = (y0, x0, y1, x1) on the target canvas, clipped.
    """
    y0, x0, y1, x1 = source_bounds
    iy, ix = divmod(edges["i"], width)
    jy, jx = divmod(edges["j"], width)
    inside = (
        (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
        & (jy >= y0) & (jy < y1) & (jx >= x0) & (jx < x1)
    )
    return inside
