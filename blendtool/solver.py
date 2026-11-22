"""Discrete gradient-domain systems: construction, solving, diagnostics.

Two genuinely different systems are built (not mere edge feathering):

1. Hard selection (Dirichlet Poisson).
   Unknowns are the pixels with mask == 1. The divergence of the guidance
   field drives a square Laplacian system; pixels adjacent to the boundary
   receive the known target colour through the right-hand side. With a
   non-empty Dirichlet boundary this discrete Laplacian is SPD. If the
   support touches/covers the whole canvas so no Dirichlet pixel exists,
   the system is singular (null space = constants); the condition is
   reported and one gauge row is appended before least-squares solving.

2. Soft selection (weighted least squares on the gradient field).
   Rows ``sqrt(w_e) (x_j - x_i - g_e)`` over all edges touching the
   support, with weights tapering through the feather band, plus anchor
   rows ``sqrt(w_a) (x_p - c_a)``. A zero-mean gauge row makes the full
   support case identifiable. Priority presets change the relative weight
   of gradient vs anchor/luminance rows.

Both paths are solved per colour channel. The module never hides the
resulting colour drift: raw float output, out-of-gamut fraction and full
residual statistics are returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, cg, lsqr, splu
import warnings

from .gradient import build_edges, guidance_field, edge_source_validity
from .masks import connected_components, holes

GAUGE_WEIGHT = 1e3
ANCHOR_WEIGHT = 50.0
LUMINANCE_WEIGHT = 0.25
LSQR_ATOL = 1e-9
LSQR_BTOL = 1e-9
LSQR_ITER_LIMIT = 20000


@dataclass
class BlendResult:
    result: np.ndarray                      # raw float HxWx3 (unclipped)
    result_display: np.ndarray              # clipped [0,1] copy for display
    mode: str
    strategy: str
    priority: str
    unknown_mask: np.ndarray                # bool HxW
    edges: dict
    guidance: np.ndarray                    # (E,3)
    edge_weights: np.ndarray               # (E,)
    row_residuals: np.ndarray               # (R,3) raw row residuals (solved channel)
    residual_map: np.ndarray                # HxW, per-pixel RMS gradient residual
    boundary_map: np.ndarray                # HxW, boundary discontinuity magnitude
    anchor_residual_map: np.ndarray         # HxW, anchor row residuals
    drift_map: np.ndarray                   # HxW, result - target (signed, luminance)
    source_delta_map: np.ndarray            # HxW, result - direct source copy
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    hole_pixels: int = 0
    anchors_used: list[dict] = field(default_factory=list)


def _check_anchors(anchors: list[dict], support: np.ndarray,
                   source_embedded: np.ndarray, target: np.ndarray) -> list[dict]:
    """Normalize anchors and detect conflicting demands on the same pixel."""
    h, w = support.shape
    out = []
    seen: dict[int, list[dict]] = {}
    for a in anchors or []:
        x, y = int(round(a["x"])), int(round(a["y"]))
        if not (0 <= x < w and 0 <= y < h) or not support[y, x]:
            continue
        color = np.asarray(a.get("color"), dtype=np.float64)
        if color.size == 1:
            color = np.repeat(color, 3)
        color = color.reshape(3)
        source_hint = a.get("source", False)
        if source_hint:
            color = source_embedded[y, x].copy()
        rec = {"x": x, "y": y, "flat": y * w + x, "color": color,
               "weight": float(a.get("weight", ANCHOR_WEIGHT)),
               "label": a.get("label", "")}
        out.append(rec)
        seen.setdefault(rec["flat"], []).append(rec)
    conflicts = []
    for flat, group in seen.items():
        if len(group) > 1:
            spread = np.ptp(np.stack([g["color"] for g in group]), axis=0).max()
            conflicts.append({"flat": int(flat), "count": len(group),
                              "max_spread": float(spread)})
    return out, conflicts


def blend(source: np.ndarray,
          target: np.ndarray,
          source_hard_mask: np.ndarray,
          offset_y: int,
          offset_x: int,
          *,
          feather_px: float = 0.0,
          strategy: str = "source",
          priority: str = "edge",
          anchors: list[dict] | None = None,
          anchor_weight: float = ANCHOR_WEIGHT,
          luminance_weight: float = LUMINANCE_WEIGHT,
          gradient_weight_scale: float = 1.0) -> BlendResult:
    """Run a full gradient-domain blend.

    source, target : float RGB arrays in [0,1]
    source_hard_mask : binary selection in source image coordinates
    offset_y/x : integer placement offset of the source onto the target
    feather_px : >0 selects soft-mask WLS mode; 0 selects hard Dirichlet
    strategy   : "source" | "mixed" | "direct"
    priority   : "edge" | "luminance" (only meaningful with anchors / soft)
    anchors    : [{"x","y" (target coords), "color":[..] or "source":True,
                   "weight": optional}]
    """
    h, w = target.shape[:2]
    hard_src = source_hard_mask.astype(bool)
    if hard_src.sum() == 0:
        raise ValueError("empty source selection")

    # Embed source and masks on the target canvas.
    src_emb = np.zeros_like(target)
    src_bounds = _embed_image(source, src_emb, offset_y, offset_x)
    hard = _embed_bool(hard_src, h, w, offset_y, offset_x)
    if hard.sum() == 0:
        raise ValueError("selection lies entirely outside the target canvas")
    soft = feather_px and float(feather_px) > 0
    from .masks import soft_mask_from_hard
    mask = soft_mask_from_hard(hard, float(feather_px)) if soft else hard.astype(np.float64)
    support = mask > 0
    unknown = support.copy()

    labels, comp_info = connected_components(hard)
    hole_mask = holes(hard)
    warnings_out: list[str] = []
    for c in comp_info:
        if c["touches_border"]:
            warnings_out.append(
                f"component {c['label']} ({c['pixels']} px) touches the canvas "
                "border; border-touching edges have no outside neighbour and "
                "use one-sided constraints")
    if hole_mask.any():
        warnings_out.append(
            f"selection contains {int(hole_mask.sum())} hole pixel(s); hole "
            "interiors keep target colours and act as interior Dirichlet regions")

    anchors_n, conflicts = _check_anchors(anchors, support, src_emb, target)
    if conflicts:
        for c in conflicts:
            warnings_out.append(
                f"{c['count']} conflicting anchors at pixel {c['flat']} "
                f"(max colour spread {c['max_spread']:.4f}); priority="
                f"'{priority}' resolves via weighted least squares")

    edges = build_edges(h, w, support)
    valid = edge_source_validity(edges, src_bounds, h, w)
    flat_i, flat_j = edges["i"], edges["j"]
    mask_flat = mask.ravel()
    g, w_edge = guidance_field(src_emb, target, edges, valid,
                               mask_flat[flat_i], mask_flat[flat_j],
                               strategy, soft)
    w_edge = w_edge * float(gradient_weight_scale)

    if strategy == "direct":
        # Direct colour copy: no gradient system is solved. Metrics are still
        # computed against the target's own gradient field so the boundary
        # cost of a raw paste is quantified rather than hidden by feathering.
        return _direct_copy(target, src_emb, hard, mask, edges, g, w_edge,
                            comp_info, int(hole_mask.sum()), warnings_out,
                            anchors_n, priority)

    # Pixel index within the unknown set.
    unknown_flat = np.flatnonzero(unknown.ravel())
    local = np.full(h * w, -1, dtype=np.int64)
    local[unknown_flat] = np.arange(unknown_flat.size)
    n = unknown_flat.size

    # All solvers use the guidance definition directly:
    #   g = v_j - v_i, edge row (x_j - x_i) = g.
    # No extra sign flip is introduced; residual reporting uses the same
    # (x_j - x_i) - g expression.
    if not soft and not anchors_n:
        x, singular, solve_info = _solve_hard(
            target, edges, g, unknown_flat, local, support, w)
        if singular:
            warnings_out.append(
                "hard-mask system is singular (no Dirichlet boundary pixel "
                "exists): a gauge constraint was added; the absolute level is "
                "not determined by the data")
    else:
        x, singular, solve_info = _solve_wls(
            target, src_emb, edges, g, w_edge, unknown_flat, local,
            support, anchors_n, priority, anchor_weight, luminance_weight, w)
        if singular:
            warnings_out.append(
                "system is singular (selection covers the entire canvas: no "
                "Dirichlet target pixels exist); a zero-mean gauge row was "
                "added and the solution is defined only up to the reported "
                "constant drift")

    result = target.copy()
    res_flat = result.reshape(-1, 3)
    res_flat[unknown_flat] = x

    br = BlendResult(
        result=result,
        result_display=np.clip(result, 0.0, 1.0),
        mode="soft_wls" if soft else "hard_dirichlet",
        strategy=strategy,
        priority=priority,
        unknown_mask=unknown,
        edges=edges,
        guidance=g,
        edge_weights=w_edge,
        row_residuals=np.zeros((0, 3)),
        residual_map=np.zeros((h, w)),
        boundary_map=np.zeros((h, w)),
        anchor_residual_map=np.zeros((h, w)),
        drift_map=np.zeros((h, w)),
        source_delta_map=np.zeros((h, w)),
        warnings=warnings_out,
        components=comp_info,
        hole_pixels=int(hole_mask.sum()),
        anchors_used=[{"x": a["x"], "y": a["y"], "color": a["color"].tolist(),
                       "weight": a["weight"], "label": a["label"]}
                      for a in anchors_n],
    )
    _finalize_metrics(br, target, src_emb, g, w_edge,
                      unknown_flat, local, hard, solve_info)
    return br


def _embed_image(source: np.ndarray, dst: np.ndarray, oy: int, ox: int) -> tuple:
    h, w = dst.shape[:2]
    sh, sw = source.shape[:2]
    y0, x0 = max(oy, 0), max(ox, 0)
    sy0, sx0 = max(-oy, 0), max(-ox, 0)
    hh = min(sh - sy0, h - y0)
    ww = min(sw - sx0, w - x0)
    if hh > 0 and ww > 0:
        dst[y0:y0 + hh, x0:x0 + ww] = source[sy0:sy0 + hh, sx0:sx0 + ww]
    return (y0, x0, y0 + hh, x0 + ww)


def _embed_bool(mask: np.ndarray, h: int, w: int, oy: int, ox: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    sh, sw = mask.shape
    y0, x0 = max(oy, 0), max(ox, 0)
    sy0, sx0 = max(-oy, 0), max(-ox, 0)
    hh = min(sh - sy0, h - y0)
    ww = min(sw - sx0, w - x0)
    if hh > 0 and ww > 0:
        out[y0:y0 + hh, x0:x0 + ww] = mask[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def _direct_copy(target, src_emb, hard, mask, edges, g, w_edge,
                 comp_info, n_holes, warnings_out, anchors, priority):
    h, w = target.shape[:2]
    result = target.copy()
    m3 = mask[..., None]
    result = target * (1.0 - m3) + src_emb * m3
    br = BlendResult(
        result=result,
        result_display=np.clip(result, 0.0, 1.0),
        mode="direct_copy",
        strategy="direct",
        priority=priority,
        unknown_mask=mask > 0,
        edges=edges,
        guidance=g,
        edge_weights=w_edge,
        row_residuals=np.zeros((0, 3)),
        residual_map=np.zeros((h, w)),
        boundary_map=np.zeros((h, w)),
        anchor_residual_map=np.zeros((h, w)),
        drift_map=np.zeros((h, w)),
        source_delta_map=np.zeros((h, w)),
        warnings=warnings_out,
        components=comp_info,
        hole_pixels=n_holes,
        anchors_used=[{"x": a["x"], "y": a["y"], "color": a["color"].tolist(),
                       "weight": a["weight"], "label": a["label"]}
                      for a in anchors],
    )
    unknown_flat = np.flatnonzero((mask > 0).ravel())
    local = np.full(h * w, -1, dtype=np.int64)
    local[unknown_flat] = np.arange(unknown_flat.size)
    _finalize_metrics(br, target, src_emb, g, w_edge, unknown_flat, local,
                      hard, {"path": "no_solve_direct_copy"})
    return br


def _solve_hard(target, edges, g, unknown_flat, local, support, width):
    """Square SPD Laplacian system with Dirichlet boundary values.

    The discrete energy sum_edges (x_j - x_i - g_e)^2 yields, for each
    unknown pixel, a degree-diagonal Laplacian row. Interior edges build
    symmetric +/-1 off-diagonal entries; boundary edges contribute only a
    diagonal term with the known target colour moved to the RHS.
    """
    i, j = edges["i"], edges["j"]
    li, lj = local[i], local[j]
    ui, uj = li >= 0, lj >= 0
    n = unknown_flat.size

    interior = ui & uj
    bnd_edges = ui ^ uj
    diag_deg = np.zeros(n)
    np.add.at(diag_deg, li[interior], 1)
    np.add.at(diag_deg, lj[interior], 1)
    bnd_unknown_local = np.where(ui[bnd_edges], li[bnd_edges], lj[bnd_edges])
    np.add.at(diag_deg, bnd_unknown_local, 1)

    # Standard discrete Poisson form (normal equations divided by the common
    # factor 2): degree on the diagonal, -1 per interior neighbour, guidance
    # divergence as RHS, and boundary target values directly on the RHS.
    off_rows = np.concatenate([li[interior], lj[interior]])
    off_cols = np.concatenate([lj[interior], li[interior]])
    off_vals = np.full(off_rows.size, -1.0)
    diag_idx = np.arange(n)
    L = sparse.coo_matrix(
        (np.concatenate([off_vals, diag_deg]),
         (np.concatenate([off_rows, diag_idx]),
          np.concatenate([off_cols, diag_idx]))),
        shape=(n, n)).tocsr()
    L.eliminate_zeros()

    # Interior divergence RHS for edge rows (x_j - x_i) = g_e.
    # Classic Poisson-fusion Dirichlet condition: an unknown pixel adjacent to
    # a fixed region takes the target value AT THAT PIXEL (x_p = t_p). The
    # boundary edge carries no source guidance; guidance acts only on interior
    # edges. This is exactly the condition that makes the rim seamless.
    tgt = target.reshape(-1, 3)
    b = np.zeros((n, 3))
    gi = g[interior]
    np.add.at(b, li[interior], -gi)
    np.add.at(b, lj[interior], gi)
    bnd_sel = np.where(bnd_edges)[0]
    for e in bnd_sel:
        u = li[e] if li[e] >= 0 else lj[e]
        u_flat = i[e] if li[e] >= 0 else j[e]
        b[u] += tgt[u_flat]

    # The Laplacian is SPD iff every connected component of the unknown graph
    # contains at least one boundary (Dirichlet) edge.
    singular = not np.any(bnd_edges)
    info = {"path": "hard_splu"}
    x = np.zeros((n, 3))
    if singular:
        info["path"] = "lsqr_gauge"
        A, bb = _gauge_system(L, b, n)
        for c in range(3):
            sol = lsqr(A, bb[:, c], atol=LSQR_ATOL, btol=LSQR_BTOL,
                       iter_lim=LSQR_ITER_LIMIT)
            x[:, c] = sol[0]
            info[f"lsqr_stop_c{c}"] = int(sol[1])
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                lu = splu(L.tocsc())
            for c in range(3):
                x[:, c] = lu.solve(b[:, c])
        except (RuntimeError, ValueError, MatrixRankWarning):
            singular = True
            info["path"] = "lsqr_gauge"
            A, bb = _gauge_system(L, b, n)
            for c in range(3):
                sol = lsqr(A, bb[:, c], atol=LSQR_ATOL, btol=LSQR_BTOL,
                           iter_lim=LSQR_ITER_LIMIT)
                x[:, c] = sol[0]
                info[f"lsqr_stop_c{c}"] = int(sol[1])
    return x, singular, info


def _solve_wls(target, src_emb, edges, g, w_edge, unknown_flat, local,
               support, anchors, priority, anchor_weight,
               luminance_weight, width):
    """Weighted least squares: gradient rows + anchor/luminance rows.

    Every row is assembled explicitly as ``sqrt(w) * (coeff dot x - rhs)``.
    Edges leaving the support collapse to a one-term row whose known endpoint
    is moved into the RHS, which is what enforces continuity through the
    feather band and at canvas borders.
    """
    i, j = edges["i"], edges["j"]
    li, lj = local[i], local[j]
    n = unknown_flat.size
    tgt = target.reshape(-1, 3)

    row_cols: list[np.ndarray] = []
    row_vals: list[np.ndarray] = []
    row_rhs: list[np.ndarray] = []
    row_sqrt: list[float] = []
    row_kind: list[str] = []

    # Edge row convention matches the guidance definition:
    #   (x_j - x_i) = g,  coefficients -1 at i, +1 at j.
    # g here is already g_solve (= -g_guidance) from the caller; the caller
    # flips sign because this assembly historically used the reverse row.
    for e in range(i.size):
        sw = np.sqrt(max(w_edge[e], 0.0))
        if sw <= 0.0:
            continue
        if li[e] >= 0 and lj[e] >= 0:
            # sqrt(w) * (x_j - x_i - g) = 0
            row_cols.append(np.array([li[e], lj[e]], dtype=np.int64))
            row_vals.append(np.array([-sw, sw]))
            row_rhs.append(sw * g[e])
        else:
            # Perez boundary row with fixed target neighbour:
            #   unknown i: x_i - t_j = -g  -> x_i = t_j - g
            #   unknown j: x_j - t_i =  g  -> x_j = t_i + g
            if li[e] >= 0:
                row_cols.append(np.array([li[e]], dtype=np.int64))
                row_vals.append(np.array([sw]))
                row_rhs.append(sw * (tgt[j[e]] - g[e]))
            else:
                row_cols.append(np.array([lj[e]], dtype=np.int64))
                row_vals.append(np.array([sw]))
                row_rhs.append(sw * (tgt[i[e]] + g[e]))
        row_sqrt.append(sw)
        row_kind.append("gradient")

    anchor_flats = []
    for a in anchors:
        lf = local[a["flat"]]
        if lf < 0:
            continue
        w = a["weight"] if a["weight"] else anchor_weight
        if priority == "luminance":
            w = w * 4.0
        sw = np.sqrt(w)
        row_cols.append(np.array([lf], dtype=np.int64))
        row_vals.append(np.array([sw]))
        row_rhs.append(sw * a["color"])
        row_sqrt.append(sw)
        row_kind.append("anchor")
        anchor_flats.append((lf, w, a["color"]))

    if priority == "luminance":
        # Soft data term pulling toward the source's RGB values (dominated by
        # luminance in practice); chroma still follows the gradient field.
        src_flat = src_emb.reshape(-1, 3)
        for lf, flat in enumerate(unknown_flat):
            sw = np.sqrt(luminance_weight)
            row_cols.append(np.array([lf], dtype=np.int64))
            row_vals.append(np.array([sw]))
            row_rhs.append(sw * src_flat[flat])
            row_sqrt.append(sw)
            row_kind.append("luminance")

    n_rows = len(row_sqrt)
    if n_rows == 0:
        raise np.linalg.LinAlgError(
            "no constraint rows: the selection has neither interior edges "
            "nor anchors (system is empty and underdetermined)")
    row_idx = np.concatenate([np.full(c.size, r, dtype=np.int64)
                              for r, c in enumerate(row_cols)])
    cols = np.concatenate(row_cols)
    vals = np.concatenate(row_vals)
    A = sparse.coo_matrix((vals, (row_idx, cols)),
                          shape=(n_rows, n)).tocsr()
    b = np.asarray(row_rhs, dtype=np.float64)
    kind = np.asarray(row_kind, dtype=object)
    sw_arr = np.asarray(row_sqrt, dtype=np.float64)

    # Singular iff no row reaches a known target value (support covers the
    # whole canvas) and no anchor pins a level.
    reaches_known = np.any((li < 0) | (lj < 0))
    singular = not reaches_known and not anchor_flats
    if singular:
        grow = sparse.coo_matrix(
            (np.full(n, np.sqrt(GAUGE_WEIGHT)),
             (np.zeros(n, dtype=np.int64), np.arange(n))),
            shape=(1, n))
        A = sparse.vstack([A, grow], format="csr")
        b = np.vstack([b, np.zeros((1, 3))])
        kind = np.concatenate([kind, np.array(["gauge"], dtype=object)])
        sw_arr = np.concatenate([sw_arr, np.full(1, np.sqrt(GAUGE_WEIGHT))])

    info = {"path": "wls_normal_cg", "rows": int(A.shape[0]),
            "unknowns": int(n), "gauge": bool(singular)}
    # Normal equations A^T A x = A^T b are SPD whenever the system has a
    # unique solution. The weighted graph Laplacian stays well conditioned
    # here (boundary rows pin every component), so plain CG converges tightly
    # where LSQR stops early on condition-number estimates.
    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b
    x = np.zeros((n, 3))
    for c in range(3):
        sol_x, code = cg(AtA, Atb[:, c], rtol=1e-12,
                         maxiter=LSQR_ITER_LIMIT)
        x[:, c] = sol_x
        info[f"cg_code_c{c}"] = int(code)
        if code != 0:
            # Fallback: high condition-limit LSQR.
            sol = lsqr(A, b[:, c], atol=LSQR_ATOL, btol=LSQR_BTOL,
                       conlim=1e14, iter_lim=LSQR_ITER_LIMIT)
            x[:, c] = sol[0]
            info[f"lsqr_stop_c{c}"] = int(sol[1])
            info["path"] = "wls_lsqr_fallback"
    return x, singular, info


def _gauge_system(L, b, n):
    grow = sparse.coo_matrix(
        (np.ones(n), (np.zeros(n, dtype=np.int64), np.arange(n))),
        shape=(1, n))
    A = sparse.vstack([L, grow], format="csr")
    bb = np.vstack([b, np.zeros((1, 3))])
    return A, bb


def _finalize_metrics(br: BlendResult, target, src_emb, guidance, edge_w,
                      unknown_flat, local, hard, solve_info):
    h, w = target.shape[:2]
    tgt = target.reshape(-1, 3)
    res = br.result.reshape(-1, 3)
    i, j = br.edges["i"], br.edges["j"]
    li, lj = local[i], local[j]

    # Realized gradients vs the requested guidance field. Boundary edges of a
    # hard selection are fixed by the Dirichlet condition, so their gradient
    # mismatch is a *boundary* quantity, not an interior solver residual.
    ui = (li >= 0)[:, None]
    uj = (lj >= 0)[:, None]
    xi = np.where(ui, res[i], tgt[i])
    xj = np.where(uj, res[j], tgt[j])
    all_res = (xj - xi) - guidance
    if br.mode == "hard_dirichlet":
        grad_fit = ((li >= 0) & (lj >= 0))
    else:
        grad_fit = np.ones(i.size, dtype=bool)
    grad_res = all_res[grad_fit]
    gr_mag = np.linalg.norm(grad_res, axis=1)

    # Per-pixel RMS gradient residual (interior fit edges only).
    fi, fj = i[grad_fit], j[grad_fit]
    rmap = np.zeros(h * w)
    count = np.zeros(h * w)
    np.add.at(rmap, fi, gr_mag ** 2)
    np.add.at(count, fi, 1)
    np.add.at(rmap, fj, gr_mag ** 2)
    np.add.at(count, fj, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmap = np.sqrt(np.where(count > 0, rmap / np.maximum(count, 1), 0.0))
    br.residual_map = rmap.reshape(h, w)
    br.row_residuals = all_res

    # Guidance mismatch on hard-mask boundary edges (cost of satisfying the
    # Dirichlet condition instead of the source gradient).
    boundary_mismatch = np.zeros(i.size)
    if br.mode == "hard_dirichlet":
        bsel = ~grad_fit
        boundary_mismatch[bsel] = np.linalg.norm(all_res[bsel], axis=1)

    # Boundary discontinuity across the hard selection rim.
    hard_flat = hard.ravel()
    cross = hard_flat[i] != hard_flat[j]
    bmap = np.zeros(h * w)
    if cross.any():
        jump = np.linalg.norm(xj[cross] - xi[cross], axis=1)
        tgt_jump = np.linalg.norm(tgt[j[cross]] - tgt[i[cross]], axis=1)
        introduced = jump - tgt_jump
        np.add.at(bmap, i[cross], np.abs(introduced))
        np.add.at(bmap, j[cross], np.abs(introduced))
    br.boundary_map = bmap.reshape(h, w)

    # Anchor residuals.
    amap = np.zeros((h, w))
    for a in br.anchors_used:
        got = res[a["y"] * w + a["x"]]
        amap[a["y"], a["x"]] = float(np.linalg.norm(got - np.asarray(a["color"])))
    br.anchor_residual_map = amap

    # Drift vs target and vs a direct (non-solved) source copy.
    br.drift_map = np.mean(br.result - target, axis=2)
    direct = np.zeros_like(target)
    direct[hard] = src_emb[hard]
    delta = br.result - direct
    delta[~hard] = 0.0
    br.source_delta_map = np.mean(delta, axis=2)

    br.metrics = _statistics(br, target, src_emb, hard, grad_res, gr_mag,
                             cross, i, j, xi, xj, res, tgt, edge_w, solve_info,
                             boundary_mismatch, grad_fit)


def _statistics(br, target, src_emb, hard, grad_res, gr_mag, cross,
                i, j, xi, xj, res, tgt, edge_w, solve_info,
                boundary_mismatch, grad_fit):
    lum = np.array([0.2126, 0.7152, 0.0722])
    inside = hard.ravel()
    m: dict = {}
    m["solver"] = solve_info
    m["mode"] = br.mode
    m["strategy"] = br.strategy
    m["priority"] = br.priority
    m["n_unknown"] = int(br.unknown_mask.sum())
    m["n_edges"] = int(i.size)
    m["n_components"] = len(br.components)
    m["n_hole_pixels"] = br.hole_pixels
    m["gradient_residual_rms"] = float(np.sqrt(np.mean(grad_res ** 2)))
    m["gradient_residual_max"] = float(np.abs(grad_res).max())
    m["gradient_residual_rms_lum"] = float(
        np.sqrt(np.mean((grad_res @ lum) ** 2)))
    m["gradient_residual_p99_lum"] = float(
        np.percentile(np.abs(grad_res @ lum), 99))
    if boundary_mismatch.any():
        m["boundary_guidance_mismatch_rms"] = float(
            np.sqrt(np.mean(boundary_mismatch[boundary_mismatch > 0] ** 2)))
    else:
        m["boundary_guidance_mismatch_rms"] = 0.0
    if br.mode == "soft_wls":
        wr = grad_res * np.sqrt(np.maximum(edge_w[grad_fit], 0.0))[:, None]
        m["weighted_residual_rms"] = float(np.sqrt(np.mean(wr ** 2)))
        m["anchor_residual_rms"] = float(
            np.sqrt(np.mean(br.anchor_residual_map[br.anchor_residual_map > 0] ** 2))
            if (br.anchor_residual_map > 0).any() else 0.0)
    if cross.any():
        jump = np.linalg.norm(xj[cross] - xi[cross], axis=1)
        tgt_jump = np.linalg.norm(tgt[j[cross]] - tgt[i[cross]], axis=1)
        m["boundary_jump_rms"] = float(np.sqrt(np.mean(jump ** 2)))
        m["boundary_introduced_rms"] = float(
            np.sqrt(np.mean((jump - tgt_jump) ** 2)))
        m["boundary_crossing_edges"] = int(cross.sum())
    else:
        m["boundary_jump_rms"] = 0.0
        m["boundary_introduced_rms"] = 0.0
        m["boundary_crossing_edges"] = 0
    drift = (res - tgt) @ lum
    d_in = drift[inside]
    m["drift_mean_lum_inside"] = float(d_in.mean())
    m["drift_std_lum_inside"] = float(d_in.std())
    m["drift_abs_max_lum"] = float(np.abs(d_in).max())
    per_ch = res[inside] - tgt[inside]
    m["drift_mean_per_channel"] = per_ch.mean(axis=0).tolist()
    m["drift_per_channel_max_abs"] = np.abs(per_ch).max(axis=0).tolist()
    m["out_of_gamut_fraction_inside"] = float(
        ((res[inside] < 0.0) | (res[inside] > 1.0)).mean())
    m["clipped_pixels_inside"] = int(
        np.any((res < 0) | (res > 1), axis=1)[inside].sum())
    # Detail retention on interior edges: realized vs source gradient energy.
    interior = inside[i] & inside[j] & grad_fit
    src_flat = src_emb.reshape(-1, 3)
    src_grad_mag = np.linalg.norm(src_flat[j] - src_flat[i], axis=1)
    got_grad_mag = np.linalg.norm(xj - xi, axis=1)
    if interior.any():
        m["source_gradient_energy_inside"] = float(
            np.sqrt(np.mean(src_grad_mag[interior] ** 2)))
        m["result_gradient_energy_inside"] = float(
            np.sqrt(np.mean(got_grad_mag[interior] ** 2)))
        m["detail_energy_ratio"] = float(
            m["result_gradient_energy_inside"]
            / max(m["source_gradient_energy_inside"], 1e-12))
    else:
        m["source_gradient_energy_inside"] = 0.0
        m["result_gradient_energy_inside"] = 0.0
        m["detail_energy_ratio"] = 0.0
    return m
