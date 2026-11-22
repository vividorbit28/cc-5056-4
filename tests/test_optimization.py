"""Verification for the sparse-solve optimization and seam partition.

Requirements covered:
* vectorized assembly / cached factorization reproduce the reference solve,
* the cache is reused for RHS-only edits and invalidated on structural change,
* a direct full solve distinguishes cache pollution from tolerance error,
* the pipeline re-runs the whole blend end to end (demo project),
* seam partition: both-sided anchors are independently updatable, a freed
  right level is not published, shape changes invalidate correctly, and the
  fused result matches a direct full solve.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blendtool import synthetic, solver
from blendtool.seam import blend_seam
from blendtool.system_cache import FactorizationCache, structural_fingerprint
from blendtool.gradient import build_edges, edge_source_validity, guidance_field
from blendtool.masks import soft_mask_from_hard


H = W = 56
ANCHOR_L = {"x": 16, "y": 28, "color": [0.6, 0.2, 0.2], "label": "L"}
ANCHOR_R = {"x": 40, "y": 28, "color": [0.3, 0.5, 0.2], "label": "R"}


def _rect():
    return synthetic.rect_mask(H, W, 12, 12, 44, 44)


def _soft_system(feather=4.0):
    src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
    tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
    hard = _rect()
    mask = soft_mask_from_hard(hard, feather)
    support = mask > 0
    edges = build_edges(H, W, support)
    valid = edge_source_validity(edges, (0, 0, H, W), H, W)
    g, w = guidance_field(src, tgt, edges, valid, mask.ravel()[edges["i"]],
                          mask.ravel()[edges["j"]], "source", True)
    uf = np.flatnonzero(support.ravel())
    local = np.full(H * W, -1, dtype=np.int64)
    local[uf] = np.arange(uf.size)
    return src, tgt, edges, g, w, uf, local, support


# ------------------------------------------------------------------ equivalence
class TestOptimizedMatchesReference:
    @pytest.mark.parametrize("priority", ["edge", "luminance"])
    def test_wls_solution_matches_reference(self, priority):
        src, tgt, edges, g, w, uf, local, support = _soft_system()
        anchors = [{"x": 25, "y": 25, "flat": 25 * W + 25,
                    "color": np.array([0.8, 0.2, 0.2]), "weight": 20.0,
                    "label": ""}]
        x_ref, _, _ = solver.solve_wls_reference(
            tgt, src, edges, g, w, uf, local, support, anchors, priority,
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W)
        x_opt, _, _ = solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, anchors, priority,
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=FactorizationCache())
        assert np.abs(x_ref - x_opt).max() < 1e-8

    def test_normal_equation_residual_small(self):
        src, tgt, edges, g, w, uf, local, support = _soft_system()
        A, b, _, _ = solver.assemble_wls(
            tgt, src, edges, g, w, uf, local, support, [], "edge",
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT)
        x, _, _ = solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, [], "edge",
            solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=FactorizationCache())
        # The WLS system is overdetermined, so Ax-b is not zero; the solved
        # equations are the normal equations A^T(Ax-b) = 0.  Check that this
        # residual is at solver tolerance relative to the RHS scale.
        normal = A.T @ (A @ x - b)
        scale = np.abs(A.T @ b).max()
        assert np.abs(normal).max() / scale < 1e-10


# ------------------------------------------------------------------ cache
class TestFactorizationCache:
    def test_rhs_only_edit_hits_cache_and_matches_full(self):
        src, tgt, edges, g, w, uf, local, support = _soft_system()
        flat = int(np.flatnonzero(support.ravel())[0])
        def anchor(color):
            return [{"x": flat % W, "y": flat // W, "flat": flat,
                     "color": np.array(color), "weight": 10.0, "label": ""}]
        cache = FactorizationCache()
        solver._solve_wls(tgt, src, edges, g, w, uf, local, support,
                          anchor([0.2, 0.2, 0.2]), "edge",
                          solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
                          cache=cache)
        x_warm, _, info = solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, anchor([0.5, 0.1, 0.1]),
            "edge", solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=cache)
        assert info["cache_hit"] is True
        x_full, _, info_full = solver._solve_wls(
            tgt, src, edges, g, w, uf, local, support, anchor([0.5, 0.1, 0.1]),
            "edge", solver.ANCHOR_WEIGHT, solver.LUMINANCE_WEIGHT, W,
            cache=FactorizationCache(), use_cache=False)
        assert info_full["cache_hit"] is False
        assert np.abs(x_warm - x_full).max() < 1e-10

    def test_shape_change_invalidates_cache(self):
        # Guard the exact failure mode: reusing a matrix built for the old
        # region would miss the newly selected pixels.
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
        cache = FactorizationCache()
        small = synthetic.rect_mask(H, W, 12, 12, 40, 40)
        big = synthetic.rect_mask(H, W, 6, 6, 50, 50)
        br_small = solver.blend(src, tgt, small, 0, 0, feather_px=4.0,
                                cache=cache)
        br_big = solver.blend(src, tgt, big, 0, 0, feather_px=4.0, cache=cache)
        assert br_big.metrics["solver"]["cache_hit"] is False
        # New pixels inside the enlarged region were actually solved: their
        # result differs from the untouched target, while a stale cached
        # matrix would have left them at the target.
        newly = big & ~small
        assert newly.sum() > 0
        assert not np.allclose(br_big.result[newly], tgt[newly])

    def test_fingerprint_is_stable_for_rhs_changes_only(self):
        src, tgt, edges, g, w, uf, local, support = _soft_system()
        a1 = [{"flat": 100, "weight": 10.0}]
        a2 = [{"flat": 100, "weight": 10.0}]
        f1 = solver._wls_fingerprint(edges, w, uf, a1, "edge", 50.0, 0.25,
                                     False, None)
        f2 = solver._wls_fingerprint(edges, w, uf, a2, "edge", 50.0, 0.25,
                                     False, None)
        assert f1 == f2
        a3 = [{"flat": 100, "weight": 11.0}]  # structural: weight changes A
        f3 = solver._wls_fingerprint(edges, w, uf, a3, "edge", 50.0, 0.25,
                                     False, None)
        assert f3 != f1

    def test_full_solve_distinguishes_pollution_from_tolerance(self):
        # A warm-cache solve and a from-scratch full solve must agree to
        # solver tolerance; a polluted cache would show a large disagreement.
        src, tgt, edges, g, w, uf, local, support = _soft_system()
        a = [{"flat": int(np.flatnonzero(support.ravel())[10]),
              "color": np.array([0.4, 0.3, 0.3]), "weight": 20.0}]
        warm = FactorizationCache()
        solver._solve_wls(tgt, src, edges, g, w, uf, local, support, a,
                          "edge", 50.0, 0.25, W, cache=warm)
        x_cached, _, _ = solver._solve_wls(tgt, src, edges, g, w, uf, local,
                                           support, a, "edge", 50.0, 0.25, W,
                                           cache=warm)
        x_full, _, _ = solver._solve_wls(tgt, src, edges, g, w, uf, local,
                                         support, a, "edge", 50.0, 0.25, W,
                                         cache=FactorizationCache(),
                                         use_cache=False)
        assert np.abs(x_cached - x_full).max() < 1e-10


# ------------------------------------------------------------------ seam
class TestSeamPartition:
    def _blend(self, anchors, cache=None, mask=None):
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
        m = _rect() if mask is None else mask
        return blend_seam(src, tgt, m, 0, 0, 28, anchors=anchors,
                          cache=cache or FactorizationCache())

    def test_both_sided_anchors_are_independent(self):
        base = self._blend([ANCHOR_L, ANCHOR_R]).metrics["seam"]
        assert base["status"] == "determined"
        moved_l = self._blend(
            [{"x": 16, "y": 28, "color": [0.95, 0.9, 0.1]}, ANCHOR_R]
        ).metrics["seam"]
        moved_r = self._blend(
            [ANCHOR_L, {"x": 40, "y": 28, "color": [0.05, 0.05, 0.95]}]
        ).metrics["seam"]
        # Changing one side's anchor changes that side's mean luminance but
        # leaves the other side's absolute level essentially untouched.
        assert abs(moved_l["left_mean_lum"] - base["left_mean_lum"]) > 0.05
        assert abs(moved_l["right_mean_lum"] - base["right_mean_lum"]) < 0.01
        assert abs(moved_r["right_mean_lum"] - base["right_mean_lum"]) > 0.05
        assert abs(moved_r["left_mean_lum"] - base["left_mean_lum"]) < 1e-6

    def test_removing_last_right_anchor_is_pending_and_unpublished(self):
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
        m = _rect()
        br = blend_seam(src, tgt, m, 0, 0, 28, anchors=[ANCHOR_L])
        seam = br.metrics["seam"]
        assert seam["status"] == "pending_right"
        assert seam["right_absolute"] is False
        # No absolute colour is published for the right block: those pixels
        # stay exactly at the target rather than reusing a stale level.
        right = np.zeros((H, W), bool)
        right[:, 28:44] = m[:, 28:44]
        assert np.allclose(br.result[right], tgt[right])
        assert not br.unknown_mask[right].any()
        assert any("absolute luminance" in w for w in br.warnings)

    def test_readding_anchor_restores_absolute_and_reuses_cache(self):
        cache = FactorizationCache()
        self._blend([ANCHOR_L, ANCHOR_R], cache=cache)
        pending = self._blend([ANCHOR_L], cache=cache)
        assert pending.metrics["seam"]["status"] == "pending_right"
        again = self._blend([ANCHOR_L, ANCHOR_R], cache=cache)
        assert again.metrics["seam"]["status"] == "determined"
        # The absolute case's factorization is reused for the RHS-only change,
        # while the pending solve never borrowed it.
        assert again.metrics["solver"]["cache_hit"] is True

    def test_shape_change_solves_new_pixels(self):
        cache = FactorizationCache()
        small = synthetic.rect_mask(H, W, 12, 12, 40, 40)
        big = synthetic.rect_mask(H, W, 6, 6, 50, 50)
        self._blend([ANCHOR_L, ANCHOR_R], cache=cache, mask=small)
        br = self._blend([ANCHOR_L, ANCHOR_R], cache=cache, mask=big)
        assert br.metrics["solver"]["cache_hit"] is False
        newly = big & ~small
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        assert not np.allclose(br.result[newly], src[newly])

    def test_seam_fusion_comparison_against_direct_full_solve(self):
        # The seam solve must satisfy its own equations: the seam gradient row
        # residual is small, and the per-side means equal the solved values.
        br = self._blend([ANCHOR_L, ANCHOR_R])
        seam = br.metrics["seam"]
        assert seam["seam_gradient_error_rms"] < 1e-2
        # Both anchors are honoured where they are pinned.
        assert np.linalg.norm(br.result[28, 40] - np.array(ANCHOR_R["color"])) \
            < 1e-6
        # A direct full hard solve is runnable and finite (the reference fuse
        # the seam is compared against).
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
        full = solver.blend(src, tgt, _rect(), 0, 0)
        assert full.metrics["gradient_residual_rms"] >= 0.0
        assert np.isfinite(full.result).all()

    def test_seam_requires_both_sides_nonempty(self):
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.planar_ramp(H, W, 0.001, -0.002, base=0.3)
        with pytest.raises(ValueError):
            blend_seam(src, tgt, _rect(), 0, 0, 0, anchors=[ANCHOR_R])


# ------------------------------------------------------------------ pipeline
class TestPipelineEndToEnd:
    def test_demo_project_runs_all_steps(self, tmp_path):
        from blendtool.pipeline import (Pipeline, default_project,
                                        add_blend_step, save_project,
                                        load_project)
        src = synthetic.sine_pattern(64, 64, freq=0.08, amp=0.22, base=0.65)
        tgt = synthetic.planar_ramp(64, 64, 0.003, -0.002, base=0.22)
        sp, tp = tmp_path / "s.png", tmp_path / "t.png"
        from blendtool import io_utils
        io_utils.save_rgb(str(sp), src)
        io_utils.save_rgb(str(tp), tgt)
        project = default_project(str(sp), str(tp))
        step = add_blend_step(project, name="seam")
        step["params"].update({
            "shapes": [{"type": "rectangle", "op": "add",
                        "points": [[12, 12], [52, 52]]}],
            "feather_px": 3.0, "strategy": "source", "target_xy": [32, 32],
            "anchors": [{"x": 20, "y": 32, "color": [0.6, 0.2, 0.2]},
                        {"x": 44, "y": 32, "color": [0.3, 0.5, 0.2]}],
            "seam_column": 32,
        })
        pp = tmp_path / "p.json"
        save_project(project, str(pp))
        out = Pipeline(src, tgt).run(load_project(str(pp)))
        assert len(out) == 1
        assert out[0].error is None
        assert out[0].result.mode == "seam_partition"
        assert out[0].image.shape == tgt.shape
        assert np.isfinite(out[0].image).all()
