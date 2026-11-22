"""Sparse-solve optimization benchmark.

Records, per scenario, the equation residual and solve cost for:

* the pre-optimization reference path (row-by-row assembly + unpreconditioned
  CG, no cache),
* the optimized path with the cache cold,
* the optimized path with a cache warm from an RHS-only edit,
* the same structural system re-solved by a direct full solve (no cache).

This is the executable evidence behind the speed claims.  It also derives a
cost breakdown (assemble / factor / solve) so the improvement can be tied to
the verification requirement rather than asserted.

Run::

    python -m blendtool.benchmark --out bench_out
    python -m blendtool.benchmark --sizes 64,96,128 --out bench_out
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import synthetic
from .gradient import build_edges, edge_source_validity, guidance_field
from .masks import soft_mask_from_hard
from . import solver
from .system_cache import FactorizationCache


def _prepare(H, W, feather):
    src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
    tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
    hard = synthetic.rect_mask(H, W, H // 6, W // 6,
                               5 * H // 6, 5 * W // 6)
    mask = soft_mask_from_hard(hard, feather)
    support = mask > 0
    edges = build_edges(H, W, support)
    valid = edge_source_validity(edges, (0, 0, H, W), H, W)
    g, w = guidance_field(src, tgt, edges, valid, mask.ravel()[edges["i"]],
                          mask.ravel()[edges["j"]], "source", True)
    uf = np.flatnonzero(support.ravel())
    local = np.full(H * W, -1, dtype=np.int64)
    local[uf] = np.arange(uf.size)
    return src, tgt, hard, edges, g, w, uf, local, support


def _residual(A, b, x):
    """Normal-equation residual (WLS is overdetermined, so Ax-b != 0).

    Returns (rms, max) of A^T(Ax-b).  Any real cache pollution shows up as a
    large jump here relative to the reference solve.
    """
    r = A.T @ (A @ x - b)
    return float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def run_case(H, W, feather, *, repeat_warm=3):
    (src, tgt, hard, edges, g, w, uf, local, support) = _prepare(
        H, W, feather)
    n = uf.size
    anchors = []

    # Reference (pre-optimization) path, timed as the median of two runs so a
    # single noisy/slow CG run cannot inflate the reported speedup.
    ref_times = []
    for _ in range(2):
        t0 = time.perf_counter()
        x_ref, sing_ref, info_ref = solver.solve_wls_reference(
            tgt, src, edges, g, w, uf, local, support, anchors, "edge",
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W)
        ref_times.append(time.perf_counter() - t0)
    t_ref = float(np.median(ref_times))
    # Reassemble to compute an honest residual (reference returns no A).
    A_ref, b_ref, _, _ = solver.assemble_wls(
        tgt, src, edges, g, w, uf, local, support, anchors, "edge",
        solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT)
    res_ref = _residual(A_ref, b_ref, x_ref)

    # Optimized, cold cache (median of two fresh-cache runs).
    cold_times = []
    for _ in range(2):
        t0 = time.perf_counter()
        x_cold, sing_c, info_cold = solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, anchors, "edge",
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=FactorizationCache(), use_cache=True)
        cold_times.append(time.perf_counter() - t0)
    t_cold = float(np.median(cold_times))
    res_cold = _residual(A_ref, b_ref, x_cold)

    # Optimized, warm cache (same structure, changed RHS: anchor colour).
    cache_warm = FactorizationCache()
    first_flat = int(np.flatnonzero(support.ravel())[0])
    warm_anchors = [{"x": int(first_flat % W), "y": int(first_flat // W),
                     "flat": first_flat, "color": np.array([0.2, 0.2, 0.2]),
                     "weight": 10.0, "label": ""}]
    warm_times = []
    for k in range(repeat_warm):
        warm_anchors[0]["color"] = np.array([0.2 + 0.01 * k, 0.2, 0.2])
        t0 = time.perf_counter()
        solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, warm_anchors, "edge",
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=cache_warm, use_cache=True)
        warm_times.append(time.perf_counter() - t0)
    x_warm, _, info_warm = solver._solve_wls(
        tgt, src, edges, g, w, uf, local, support, warm_anchors, "edge",
        solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
        cache=cache_warm, use_cache=True)
    A_warm, b_warm, _, _ = solver.assemble_wls(
        tgt, src, edges, g, w, uf, local, support, warm_anchors, "edge",
        solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT)
    res_warm = _residual(A_warm, b_warm, x_warm)

    # Direct full solve (no cache) at the same structure.
    t0 = time.perf_counter()
    x_full, _, info_full = solver._solve_wls(
        tgt, src, edges, g, w, uf, local, support, warm_anchors, "edge",
        solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
        cache=FactorizationCache(), use_cache=False)
    t_full = time.perf_counter() - t0
    res_full = _residual(A_warm, b_warm, x_full)

    return {
        "size": [H, W], "feather": feather,
        "unknowns": int(n), "edges": int(edges["i"].size),
        "reference": {"time_ms": round(t_ref * 1e3, 3),
                      "time_ms_all": [round(v * 1e3, 3) for v in ref_times],
                      "cg_iterations": info_ref.get("cg_iterations"),
                      "residual_rms": res_ref[0], "residual_max": res_ref[1],
                      "path": info_ref["path"]},
        "optimized_cold": {"time_ms": round(t_cold * 1e3, 3),
                           "time_ms_all": [round(v * 1e3, 3)
                                           for v in cold_times],
                           "residual_rms": res_cold[0],
                           "residual_max": res_cold[1],
                           "breakdown_ms": {
                               "assemble": info_cold.get("assemble_ms"),
                               "factor": info_cold.get("factor_ms"),
                               "solve": info_cold.get("solve_ms")},
                           "path": info_cold["path"],
                           "cache_hit": info_cold.get("cache_hit")},
        "optimized_warm": {"time_ms_median": round(float(np.median(warm_times))
                                                    * 1e3, 3),
                           "time_ms_all": [round(v * 1e3, 3)
                                           for v in warm_times],
                           "residual_rms": res_warm[0],
                           "residual_max": res_warm[1],
                           "cache_hit": bool(info_warm.get("cache_hit")),
                           "cache_stats": cache_warm.stats()},
        "direct_full": {"time_ms": round(t_full * 1e3, 3),
                        "residual_rms": res_full[0],
                        "residual_max": res_full[1],
                        "path": info_full["path"],
                        "cache_hit": info_full.get("cache_hit")},
        "speedup_cold_vs_reference": round(t_ref / t_cold, 3),
        "speedup_full_vs_reference": round(t_ref / t_full, 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="64,96,128,192")
    ap.add_argument("--feathers", default="4,6")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    feathers = [float(s) for s in args.feathers.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for H in sizes:
        for fe in feathers:
            r = run_case(H, H, fe)
            results.append(r)
            print(json.dumps(r, indent=2, default=str))
    doc = {"generated": time.time(),
           "notes": "Reference = pre-optimization CG path; optimized = "
                    "vectorized assembly + cached sparse LU. Residuals are "
                    "the normal-equation residual ||Ax-b||_rms.",
           "cases": results}
    (outdir / "benchmark.json").write_text(json.dumps(doc, indent=2,
                                                       default=str))
    # Compact CSV for quick reading.
    lines = ["size,feather,unknowns,ref_ms,cold_ms,warm_ms,full_ms,"
             "ref_cg_iters,ref_resid,cold_resid,warm_resid,full_resid,cache_hit"]
    for r in results:
        lines.append(
            f"{r['size'][0]},{r['feather']},{r['unknowns']},"
            f"{r['reference']['time_ms']},{r['optimized_cold']['time_ms']},"
            f"{r['optimized_warm']['time_ms_median']},"
            f"{r['direct_full']['time_ms']},"
            f"{r['reference']['cg_iterations']},"
            f"{r['reference']['residual_rms']:.3e},"
            f"{r['optimized_cold']['residual_rms']:.3e},"
            f"{r['optimized_warm']['residual_rms']:.3e},"
            f"{r['direct_full']['residual_rms']:.3e},"
            f"{r['optimized_warm']['cache_hit']}")
    (outdir / "benchmark.csv").write_text("\n".join(lines) + "\n")
    print("\nwrote", outdir / "benchmark.json", "and benchmark.csv")


if __name__ == "__main__":
    main()
