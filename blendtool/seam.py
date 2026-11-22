"""Seam-partitioned blend: cut the region in two and solve each side's level
separately.

Model
-----
A vertical seam at ``seam_column`` splits the support into a left block L and
a right block R.

* L is solved with absolute rows: interior gradient rows, target-pinned
  boundary rows for the part of L's rim that faces the target (Dirichlet
  rim), and any anchors inside L.  L therefore has an absolute luminance
  datum in the normal case.
* R is solved with interior gradient rows and *no* target-pinned rim rows
  ("right side constrained only by gradients and the relative boundary
  difference").  Each seam edge contributes a relative row
  ``x_R - x_L - c = g_seam`` with one shared free scalar ``c``: this single
  offset is the right block's only absolute luminance datum.
* An anchor inside R pins ``c`` (through the seam rows) and therefore pins
  R's absolute level.  With a luminance anchor on each side both blocks are
  independently determined.
* Remove the last anchor on R and ``c`` becomes free: gradients and seam
  differences still fix R's *relative* luminance, but not its absolute
  level.  The solve is then reported ``pending_right`` and no absolute colour
  is published for R.  The editor can either keep the partition pending or
  add an anchor to R and re-solve.

Cache correctness
-----------------
The factorization fingerprint includes the support geometry, the left/right
split, the kept edge set and the anchor positions/weights.  Removing the
right anchor changes the fingerprint, so a factorization built while the
anchor existed can never be reused to publish a stale absolute level.  When
the right block is pending its pixels are left at the target instead of
being overwritten.

Acceptance metrics
------------------
``metrics["seam"]`` reports the per-side mean luminance, the realized seam
gradient and its error against the requested gradient, and the partition's
absolute/pending status, so the export can be checked on both quantities.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

from . import solver as _solver
from .gradient import build_edges, guidance_field, edge_source_validity
from .masks import connected_components as mask_components, holes
from .system_cache import DEFAULT_CACHE, CachedSystem, structural_fingerprint

SEAM_MODE = "seam_partition"
_LUM = np.array([0.2126, 0.7152, 0.0722])


def _partition_analysis(right: np.ndarray, edges, cross: np.ndarray,
                        anchor_flats: set[int], local) -> dict:
    """Per right-block component, decide whether its free constant is pinned.

    Nodes are right-block pixels, joined by right-internal edges only.  A
    component is tied to the shared offset ``c`` if it touches a seam edge.
    ``c`` is pinned iff some seam-touching component contains an anchor.
    """
    r = right.ravel()
    rnodes = np.flatnonzero(r)
    if rnodes.size == 0:
        return {"exists": False, "absolute": False, "c_free": False,
                "free_components": [], "components": 0}
    label = np.full(r.size, -1, dtype=np.int64)
    label[rnodes] = np.arange(rnodes.size)
    i, j = edges["i"], edges["j"]
    both_unknown = (local[i] >= 0) & (local[j] >= 0)
    internal = both_unknown & r[i] & r[j] & ~cross
    src = label[i[internal]]
    dst = label[j[internal]]
    n = rnodes.size
    graph = sparse.coo_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    ncomp = int(comp.max()) + 1 if n else 0
    anchored = np.zeros(ncomp, dtype=bool)
    seam = np.zeros(ncomp, dtype=bool)
    for flat in anchor_flats:
        if r[flat]:
            anchored[comp[label[flat]]] = True
    for e in np.flatnonzero(cross):
        for flat in (i[e], j[e]):
            if r[flat]:
                seam[comp[label[flat]]] = True
                break
    free_components = [int(k) for k in range(ncomp)
                       if not anchored[k] and not seam[k]]
    c_free = bool(seam.any() and not (anchored & seam).any())
    absolute = (bool(seam.any()) and not c_free
                and not free_components)
    if not seam.any():
        # Right side disconnected from the seam: every component must carry
        # its own anchor to be absolute.
        absolute = bool(ncomp) and not free_components and bool(anchored.all())
    return {"exists": True, "components": int(ncomp),
            "free_components": free_components, "c_free": c_free,
            "absolute": bool(absolute),
            "anchored_components": int(anchored.sum()),
            "seam_components": int(seam.sum())}


def _fingerprint(*, support, left, right, edges, kept, w_edge, anchors,
                 priority, anchor_weight, luminance_weight, use_offset):
    anchor_pos = tuple(sorted((int(a["flat"]),
                               float(a["weight"] if a["weight"]
                                     else anchor_weight))
                              for a in anchors))
    return structural_fingerprint(
        kind="seam", support=support, left=left, right=right,
        edges_i=edges["i"][kept], edges_j=edges["j"][kept],
        w=np.round(np.asarray(w_edge, dtype=np.float64)[kept], 12),
        anchors=anchor_pos, priority=str(priority),
        anchor_weight=float(anchor_weight),
        luminance_weight=(float(luminance_weight)
                          if priority == "luminance" else -1.0),
        offset=bool(use_offset))


def blend_seam(source: np.ndarray,
               target: np.ndarray,
               source_hard_mask: np.ndarray,
               offset_y: int,
               offset_x: int,
               seam_column: int,
               *,
               feather_px: float = 0.0,
               strategy: str = "source",
               priority: str = "edge",
               anchors: list[dict] | None = None,
               anchor_weight: float = _solver.ANCHOR_WEIGHT,
               luminance_weight: float = _solver.LUMINANCE_WEIGHT,
               gradient_weight_scale: float = 1.0,
               cache=None,
               use_cache: bool = True):
    """Blend with a vertical seam at target-canvas column ``seam_column``."""
    cache = DEFAULT_CACHE if cache is None else cache
    t_start = time.perf_counter()
    h, w = target.shape[:2]
    hard_src = source_hard_mask.astype(bool)
    if hard_src.sum() == 0:
        raise ValueError("empty source selection")

    src_emb = np.zeros_like(target)
    src_bounds = _solver._embed_image(source, src_emb, offset_y, offset_x)
    hard = _solver._embed_bool(hard_src, h, w, offset_y, offset_x)
    if hard.sum() == 0:
        raise ValueError("selection lies entirely outside the target canvas")
    soft = feather_px and float(feather_px) > 0
    from .masks import soft_mask_from_hard
    mask = (soft_mask_from_hard(hard, float(feather_px)) if soft
            else hard.astype(np.float64))
    support = mask > 0

    _, comp_info = mask_components(hard)
    hole_mask = holes(hard)
    warnings: list[str] = []
    for c in comp_info:
        if c["touches_border"]:
            warnings.append(
                f"component {c['label']} ({c['pixels']} px) touches the canvas "
                "border; border edges use one-sided constraints")
    if hole_mask.any():
        warnings.append(
            f"selection contains {int(hole_mask.sum())} hole pixel(s); they "
            "keep target colours")

    anchors_n, conflicts = _solver._check_anchors(
        anchors, support, src_emb, target)
    for c in conflicts:
        warnings.append(
            f"{c['count']} conflicting anchors at pixel {c['flat']} "
            f"(max colour spread {c['max_spread']:.4f})")

    edges = build_edges(h, w, support)
    valid = edge_source_validity(edges, src_bounds, h, w)
    mask_flat = mask.ravel()
    g, w_edge = guidance_field(src_emb, target, edges, valid,
                               mask_flat[edges["i"]], mask_flat[edges["j"]],
                               strategy, soft)
    w_edge = w_edge * float(gradient_weight_scale)

    xs = np.arange(w)[None, :]
    left = support & (xs < int(seam_column))
    right = support & ~left
    if left.sum() == 0 or right.sum() == 0:
        raise ValueError(
            f"seam column {seam_column} leaves one side empty "
            f"(left={int(left.sum())} px, right={int(right.sum())} px)")

    unknown_flat = np.flatnonzero(support.ravel())
    local = np.full(h * w, -1, dtype=np.int64)
    local[unknown_flat] = np.arange(unknown_flat.size)
    n = unknown_flat.size
    lf, rf = left.ravel(), right.ravel()

    i, j = edges["i"], edges["j"]
    li, lj = local[i], local[j]
    both_unknown = (li >= 0) & (lj >= 0)
    one_unknown = (li >= 0) ^ (lj >= 0)
    cross = both_unknown & (lf[i] != lf[j])
    boundary_l = one_unknown & (lf[i] | lf[j])
    boundary_r = one_unknown & (rf[i] | rf[j])
    # Edges eligible for the weight threshold.
    usable = w_edge > 0.0

    anchor_flats = {a["flat"] for a in anchors_n}
    left_anchors = [a for a in anchors_n if lf[a["flat"]]]
    right_anchors = [a for a in anchors_n if rf[a["flat"]]]
    part = _partition_analysis(right, edges, cross, anchor_flats, local)

    left_absolute = bool(boundary_l.any()) or bool(left_anchors)
    right_absolute = bool(part.get("absolute", False))
    status = "determined" if right_absolute else "pending_right"

    if not left_absolute:
        warnings.append(
            "left block has neither target boundary nor anchor; a gauge row "
            "fixes the arbitrary constant")
    if not right_absolute:
        warnings.append(
            "right partition has no absolute luminance datum: gradients and "
            "seam differences fix only relative luminance. No absolute colour "
            "is published for the right side; add an anchor on the right and "
            "re-solve, or keep it pending.")
        if part.get("free_components"):
            warnings.append(
                f"right side has {len(part['free_components'])} disconnected "
                "block(s) with neither anchor nor seam edge")

    # ---- choose which rows/unknowns participate.
    use_offset = bool(right_absolute and cross.any())
    solve_right = right_absolute
    keep = usable.copy()
    keep &= ~boundary_r                     # right rim never target-pinned
    if not solve_right:
        keep &= ~(both_unknown & rf[i] & rf[j])   # no right interior rows
        keep &= ~cross                           # no seam rows
    # When the right side is pending we solve a reduced left-only system so no
    # all-zero (unpinnable) right column enters the matrix.
    solved = support if solve_right else left
    solved_flat = np.flatnonzero(solved.ravel())
    col_of = np.full(h * w, -1, dtype=np.int64)
    col_of[solved_flat] = np.arange(solved_flat.size)
    nsolved = solved_flat.size
    ncols = nsolved + (1 if use_offset else 0)

    row_blocks = []   # (rows, cols, vals)
    rhs_blocks = []   # (start_row, rhs (k,3))
    row_ptr = 0

    def add(rows, cols, vals, rhs=None):
        nonlocal row_ptr
        k = int(rows.max()) + 1 if rows.size else 0
        row_blocks.append((rows, cols, vals))
        if rhs is not None:
            rhs_blocks.append((row_ptr, rhs))
        row_ptr += k

    ci = col_of[i]
    cj = col_of[j]
    c_interior = (ci >= 0) & (cj >= 0)
    c_one = (ci >= 0) ^ (cj >= 0)
    c_cross = c_interior & (lf[i] != lf[j])

    # Left + right interior gradient rows.
    interior = c_interior & ~c_cross & keep
    idx = np.flatnonzero(interior)
    if idx.size:
        sw = np.sqrt(w_edge[idx])
        rows = np.repeat(np.arange(row_ptr, row_ptr + idx.size), 2)
        cols = np.stack([ci[idx], cj[idx]], axis=1).ravel()
        vals = np.stack([-sw, sw], axis=1).ravel()
        add(rows, cols, vals, sw[:, None] * g[idx])

    # Seam rows: x_R - x_L - c = g_signed, weighted.
    idx = np.flatnonzero(c_cross & keep)
    if idx.size:
        sw = np.sqrt(w_edge[idx])
        r_local = np.where(rf[i[idx]], ci[idx], cj[idx])
        l_local = np.where(lf[i[idx]], ci[idx], cj[idx])
        # g = v[j] - v[i]; express as x_R - x_L = g_signed.
        same_orientation = rf[i[idx]][:, None]   # i is the right endpoint
        g_signed = np.where(same_orientation, -g[idx], g[idx])
        if use_offset:
            rows = np.repeat(np.arange(row_ptr, row_ptr + idx.size), 3)
            cols = np.stack([r_local, l_local,
                             np.full(idx.size, nsolved)], axis=1).ravel()
            vals = np.stack([sw, -sw, -sw], axis=1).ravel()
        else:
            rows = np.repeat(np.arange(row_ptr, row_ptr + idx.size), 2)
            cols = np.stack([r_local, l_local], axis=1).ravel()
            vals = np.stack([sw, -sw], axis=1).ravel()
        add(rows, cols, vals, sw[:, None] * g_signed)

    # Left boundary rows (target-pinned Dirichlet rim).
    idx = np.flatnonzero(c_one & keep & (ci >= 0))
    if idx.size:
        sw = np.sqrt(w_edge[idx])
        unk = np.where(ci[idx] >= 0, ci[idx], cj[idx])
        fixed_flat = np.where(ci[idx] >= 0, j[idx], i[idx])
        sign = np.where(ci[idx] >= 0, 1.0, -1.0)
        rhs = (target.reshape(-1, 3)[fixed_flat]
               - sign[:, None] * g[idx])
        add(np.arange(row_ptr, row_ptr + idx.size, dtype=np.int64),
            unk.astype(np.int64), sw, sw[:, None] * rhs)
    # Right boundary rows for a disconnected absolute right block are still
    # omitted by design; an anchor supplies its datum.

    # Anchor rows (both sides; only columns present in the reduced system).
    for a in anchors_n:
        col = col_of[a["flat"]]
        if col < 0:
            continue
        wt = a["weight"] if a["weight"] else anchor_weight
        if priority == "luminance":
            wt *= 4.0
        s = np.sqrt(wt)
        add(np.array([row_ptr], dtype=np.int64),
            np.array([col], dtype=np.int64), np.array([s]),
            s * np.asarray(a["color"])[None, :])

    # Luminance data rows: left block only (never fabricate an absolute level
    # for an unanchored right block).
    if priority == "luminance":
        cols_l = col_of[np.flatnonzero(lf)]
        cols_l = cols_l[cols_l >= 0]
        if cols_l.size:
            s = np.sqrt(luminance_weight)
            add(np.arange(row_ptr, row_ptr + cols_l.size, dtype=np.int64),
                cols_l.astype(np.int64), np.full(cols_l.size, s),
                s * src_emb.reshape(-1, 3)[solved_flat[cols_l]])

    gauge_added = False
    if not left_absolute and row_ptr > 0:
        add(np.full(nsolved, row_ptr, dtype=np.int64),
            np.arange(nsolved, dtype=np.int64),
            np.full(nsolved, np.sqrt(_solver.GAUGE_WEIGHT)),
            np.zeros((nsolved, 3)))
        gauge_added = True

    if row_ptr == 0:
        raise ValueError("seam system has no constraint rows")

    rows = np.concatenate([r for r, _, _ in row_blocks])
    cols = np.concatenate([c for _, c, _ in row_blocks])
    vals = np.concatenate([v for _, _, v in row_blocks])
    A = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(row_ptr, ncols)).tocsr()
    A.eliminate_zeros()
    b = np.zeros((row_ptr, 3))
    for start, rhsv in rhs_blocks:
        b[start:start + rhsv.shape[0]] = rhsv

    fp = _fingerprint(support=support, left=left, right=right, edges=edges,
                      kept=keep, w_edge=w_edge, anchors=anchors_n,
                      priority=priority, anchor_weight=anchor_weight,
                      luminance_weight=luminance_weight, use_offset=use_offset)
    entry = cache.get(fp) if use_cache else None
    reused = entry is not None and entry.A.shape == A.shape
    t0 = time.perf_counter()
    if reused:
        lu = entry.lu
    else:
        AtA = (A.T @ A).tocsc()
        lu = splu(AtA)
        if use_cache:
            cache.put(CachedSystem(fingerprint=fp, A=A, AtA=AtA, lu=lu,
                                   meta={"rows": int(A.shape[0]),
                                         "status": status}))
    t_factor = time.perf_counter() - t0
    t0 = time.perf_counter()
    Atb = A.T @ b
    x = np.zeros((ncols, 3))
    for ch in range(3):
        x[:, ch] = lu.solve(Atb[:, ch])
    t_solve = time.perf_counter() - t0

    result = target.copy()
    res_flat = result.reshape(-1, 3)
    res_flat[solved_flat] = x[:nsolved]
    if not solve_right:
        # Pending: right side is exactly the target, never a stale solve.
        res_flat[rf] = target.reshape(-1, 3)[rf]
    published = support.copy() if solve_right else left.copy()

    solve_info = {
        "path": "seam_cached_splu" if reused else "seam_splu",
        "rows": int(A.shape[0]), "unknowns": int(n), "gauge": gauge_added,
        "cache_hit": bool(reused), "factor_ms": round(t_factor * 1e3, 3),
        "solve_ms": round(t_solve * 1e3, 3), "status": status,
        "right_absolute": right_absolute,
        "offset_variable": use_offset,
    }
    if use_offset:
        solve_info["offset_c"] = x[nsolved].tolist()

    br = _solver.BlendResult(
        result=result, result_display=np.clip(result, 0.0, 1.0),
        mode=SEAM_MODE, strategy=strategy, priority=priority,
        unknown_mask=published, edges=edges, guidance=g, edge_weights=w_edge,
        row_residuals=np.zeros((0, 3)), residual_map=np.zeros((h, w)),
        boundary_map=np.zeros((h, w)), anchor_residual_map=np.zeros((h, w)),
        drift_map=np.zeros((h, w)), source_delta_map=np.zeros((h, w)),
        warnings=warnings, components=comp_info,
        hole_pixels=int(hole_mask.sum()),
        anchors_used=[{"x": a["x"], "y": a["y"],
                       "color": a["color"].tolist(), "weight": a["weight"],
                       "label": a["label"],
                       "block": ("left" if lf[a["flat"]] else "right")}
                      for a in anchors_n],
    )
    _solver._finalize_metrics(br, target, src_emb, g, w_edge,
                              unknown_flat, local, hard, solve_info)

    lum_res = result.reshape(-1, 3) @ _LUM
    left_vals = lum_res[lf]
    right_vals = lum_res[rf]
    cross_idx = np.flatnonzero(cross)
    seam_err = 0.0
    seam_real = 0.0
    seam_src_err = 0.0
    seam_offset = None
    if cross_idx.size:
        xi = result.reshape(-1, 3)[i[cross_idx]]
        xj = result.reshape(-1, 3)[j[cross_idx]]
        realized = xj - xi
        same_orientation = rf[i[cross_idx]][:, None]
        # Requested seam difference in the x_R - x_L orientation.
        g_signed = np.where(same_orientation, -g[cross_idx], g[cross_idx])
        seam_real = float(np.sqrt(np.mean(np.sum(realized ** 2, axis=1))))
        if use_offset:
            # The row is x_R - x_L - c = g.  The offset c (the right block's
            # honest luminance datum) is part of the solution, so the equation
            # residual is realized - g - c; the seam gradient itself is
            # measured against the *source* gradient of the seam edges.
            c = x[nsolved]
            seam_offset = c.tolist()
            seam_err = float(np.sqrt(np.mean(
                np.sum((realized - g_signed - c[None, :]) ** 2, axis=1))))
            src_flat = src_emb.reshape(-1, 3)
            src_d = src_flat[j[cross_idx]] - src_flat[i[cross_idx]]
            seam_src_err = float(np.sqrt(np.mean(
                np.sum((realized - src_d) ** 2, axis=1))))
        else:
            seam_err = float(np.sqrt(np.mean(
                np.sum((realized - g_signed) ** 2, axis=1))))

    def _seam_jump_rms(arr):
        a_flat = arr.reshape(-1)
        diffs = np.array([a_flat[j[e]] - a_flat[i[e]] for e in cross_idx])
        return float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else 0.0

    seam_jump_lum = _seam_jump_rms(lum_res)
    seam_jump_rgb = [_seam_jump_rms(result[..., c]) for c in range(3)]

    m = br.metrics
    m["seam"] = {
        "column": int(seam_column),
        "left_pixels": int(left.sum()), "right_pixels": int(right.sum()),
        "left_anchors": len(left_anchors), "right_anchors": len(right_anchors),
        "left_absolute": bool(left_absolute),
        "right_absolute": bool(right_absolute),
        "status": status,
        "offset_c": seam_offset,
        "left_mean_lum": float(left_vals.mean()),
        "right_mean_lum": float(right_vals.mean()),
        "seam_gradient_rms": seam_real,
        "seam_gradient_error_rms": seam_err,
        "seam_source_gradient_error_rms": seam_src_err,
        "seam_visible_jump_rms_lum": seam_jump_lum,
        "seam_visible_jump_rms_rgb": seam_jump_rgb,
        "seam_crossing_edges": int(cross_idx.size),
        "partition": part,
    }
    m["left_mean_lum"] = float(left_vals.mean())
    m["right_mean_lum"] = float(right_vals.mean())
    m["seam_gradient_rms"] = seam_real
    m["seam_gradient_error_rms"] = seam_err
    m["seam_status"] = status
    br.metrics = m
    br.system = {"A": A, "b": b, "unknown_flat": unknown_flat, "local": local,
                 "seam_column": int(seam_column), "offset_used": use_offset}
    m["solver"]["total_ms"] = round(
        (time.perf_counter() - t_start) * 1e3, 3)
    return br
