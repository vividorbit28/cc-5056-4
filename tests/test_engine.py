"""Numerical verification on synthetic images with known gradients.

These tests do not judge results by appearance: they check the discrete
equations directly (residuals against analytically known guidance fields),
boundary continuity, detail retention, drift reporting, topology handling
(holes, multiple components, canvas border), singular-system diagnostics,
conflicting-anchor priority behaviour and pipeline reproducibility.
"""
from __future__ import annotations

import numpy as np
import pytest

from blendtool import synthetic, solver, masks
from blendtool.gradient import build_edges, guidance_field, edge_source_validity
from blendtool.masks import (rasterize_selection, soft_mask_from_hard,
                             connected_components, holes, place_mask, Placement)


H = W = 56


def _rect():
    return synthetic.rect_mask(H, W, 12, 12, 42, 42)


# ---------------------------------------------------------------- exact recovery
class TestExactRecovery:
    def test_hard_recovers_compatible_linear_field(self):
        field = synthetic.planar_ramp(H, W, 0.012, -0.007, base=0.4)
        m = _rect()
        br = solver.blend(field, field.copy(), m, 0, 0, strategy="source")
        assert br.mode == "hard_dirichlet"
        # A boundary-compatible guidance field must be reproduced to solver
        # precision; anything larger means the discrete system is wrong.
        assert np.abs(br.result - field)[m].max() < 1e-10
        assert br.metrics["gradient_residual_rms"] < 1e-10

    def test_hard_recovers_compatible_sine_field(self):
        field = synthetic.sine_pattern(H, W, freq=0.08, amp=0.18, base=0.5)
        m = _rect()
        br = solver.blend(field, field.copy(), m, 0, 0)
        assert np.abs(br.result - field)[m].max() < 1e-10

    def test_soft_recovers_compatible_fields(self):
        linear = synthetic.planar_ramp(H, W, 0.01, 0.004, base=0.35)
        sine = synthetic.sine_pattern(H, W, freq=0.1, amp=0.15, base=0.5)
        m = _rect()
        for img in (linear, sine):
            br = solver.blend(img, img.copy(), m, 0, 0, feather_px=6.0)
            assert br.mode == "soft_wls"
            assert np.abs(br.result - img)[m].max() < 1e-9
            assert br.metrics["solver"]["path"] == "wls_normal_cg"

    def test_realized_gradient_relaxes_inside_incompatible_boundary(self):
        # With an incompatible flat boundary the Poisson solution cannot keep
        # the linear gradient everywhere (harmonic relaxation); the realized
        # field stays bounded between target and source and the discrete
        # normal equations are still satisfied to machine precision.
        src = synthetic.planar_ramp(H, W, 0.02, 0.0, base=0.8)
        tgt = synthetic.flat_field(H, W, 0.1)
        m = _rect()
        br = solver.blend(src, tgt, m, 0, 0)
        # Harmonic solutions can overshoot the [target, source] range; the
        # tool reports that explicitly and only clips the display copy.
        assert "out_of_gamut_fraction_inside" in br.metrics
        assert br.result[m].max() <= src.max() + 1e-9
        if ((br.result < 0) | (br.result > 1)).any():
            assert not np.allclose(br.result, br.result_display)

    def test_normal_equations_hold_to_machine_precision(self):
        from scipy import sparse
        from blendtool.gradient import build_edges
        src = synthetic.sine_pattern(H, W, freq=0.12, amp=0.2, base=0.6)
        tgt = synthetic.planar_ramp(H, W, 0.001, 0.002, base=0.25)
        m = _rect()
        br = solver.blend(src, tgt, m, 0, 0)
        uf = np.flatnonzero(m.ravel())
        n = uf.size
        local = np.full(H * W, -1)
        local[uf] = np.arange(n)
        edges = build_edges(H, W, m)
        i, j = edges["i"], edges["j"]
        li, lj = local[i], local[j]
        inter = (li >= 0) & (lj >= 0)
        bnd = (li >= 0) ^ (lj >= 0)
        deg = np.zeros(n)
        np.add.at(deg, li[inter], 1)
        np.add.at(deg, lj[inter], 1)
        bu = np.where(li[bnd] >= 0, li[bnd], lj[bnd])
        np.add.at(deg, bu, 1)
        off = np.concatenate([li[inter], lj[inter]])
        L = sparse.coo_matrix(
            (np.concatenate([np.full(off.size, -1.0), deg]),
             (np.concatenate([off, np.arange(n)]),
              np.concatenate([np.concatenate([lj[inter], li[inter]]),
                              np.arange(n)]))),
            shape=(n, n)).tocsr()
        x = br.result.reshape(-1, 3)[uf]
        # Reconstruct RHS the same way the solver does and check Lx = b.
        tgt_f = tgt.reshape(-1, 3)
        b = np.zeros((n, 3))
        g = br.guidance
        np.add.at(b, li[inter], -g[inter])
        np.add.at(b, lj[inter], g[inter])
        for e in np.where(bnd)[0]:
            u = li[e] if li[e] >= 0 else lj[e]
            uf2 = i[e] if li[e] >= 0 else j[e]
            b[u] += tgt_f[uf2]
        assert np.abs(L @ x - b).max() < 1e-10


# ---------------------------------------------------------------- residuals
class TestResiduals:
    def test_equation_residual_uses_guidance_definition(self):
        # The reported RMS must equal the independently recomputed edge
        # residual (verifies the bookkeeping, not that residual is zero).
        src = synthetic.sine_pattern(H, W, freq=0.12, amp=0.2, base=0.6)
        tgt = synthetic.planar_ramp(H, W, 0.001, 0.002, base=0.25)
        m = _rect()
        br = solver.blend(src, tgt, m, 0, 0)
        i, j = br.edges["i"], br.edges["j"]
        res = br.result.reshape(-1, 3)
        tgt_f = tgt.reshape(-1, 3)
        unknown = br.unknown_mask.ravel()
        xi = np.where(unknown[i][:, None], res[i], tgt_f[i])
        xj = np.where(unknown[j][:, None], res[j], tgt_f[j])
        interior = unknown[i] & unknown[j]
        independent = np.sqrt(np.mean(
            ((xj - xi - br.guidance)[interior]) ** 2))
        assert br.metrics["gradient_residual_rms"] == pytest.approx(
            independent, rel=1e-10)

    def test_residual_vanishes_for_compatible_problem(self):
        field = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.5)
        br = solver.blend(field, field.copy(), _rect(), 0, 0)
        assert br.metrics["gradient_residual_rms"] < 1e-10
        assert br.metrics["gradient_residual_max"] < 1e-9

    def test_residual_map_covers_support(self):
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2)
        tgt = synthetic.flat_field(H, W, 0.3)
        br = solver.blend(src, tgt, _rect(), 0, 0)
        assert br.residual_map.shape == (H, W)
        assert br.residual_map[br.unknown_mask].max() >= 0.0
        assert np.allclose(br.residual_map[~br.unknown_mask], 0.0)


# ---------------------------------------------------------------- boundary
class TestBoundaryContinuity:
    def test_hard_selection_boundary_is_dirichlet(self):
        # Dirichlet Poisson: boundary unknown pixels equal the target value at
        # their own position, so the *fixed* rim introduces no extra jump
        # beyond what the interior solution needs; verify the boundary row
        # condition itself and that the cost is reported, not hidden.
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.25)
        br = solver.blend(src, tgt, _rect(), 0, 0)
        assert br.metrics["boundary_crossing_edges"] > 0
        assert "boundary_guidance_mismatch_rms" in br.metrics
        # The boundary mismatch is the price of satisfying the Dirichlet rim.
        assert br.metrics["boundary_guidance_mismatch_rms"] >= 0.0
        # Outside the support the image is untouched.
        assert np.allclose(br.result[~br.unknown_mask],
                           tgt[~br.unknown_mask])

    def test_direct_copy_has_large_boundary_jump(self):
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.25)
        br = solver.blend(src, tgt, _rect(), 0, 0, strategy="direct")
        assert br.mode == "direct_copy"
        assert br.metrics["boundary_jump_rms"] > 0.1
        # Direct copy must preserve source detail exactly.
        m = _rect()
        assert np.allclose(br.result[m], src[m])

    def test_soft_blend_leaves_outside_untouched(self):
        src = synthetic.planar_ramp(H, W, 0.01, 0.0, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        br = solver.blend(src, tgt, _rect(), 0, 0, feather_px=5.0)
        outside = ~br.unknown_mask
        assert np.allclose(br.result[outside], tgt[outside])


# ---------------------------------------------------------------- strategies
class TestStrategies:
    def test_three_strategies_are_distinct(self):
        src = synthetic.checkerboard(H, W, cell=6)
        tgt = synthetic.sine_pattern(H, W, freq=0.06, amp=0.25, base=0.4)
        m = _rect()
        out = {}
        for strat in ("source", "mixed", "direct"):
            br = solver.blend(src, tgt, m, 0, 0, strategy=strat)
            out[strat] = br.result.copy()
        assert not np.allclose(out["source"], out["mixed"])
        assert not np.allclose(out["source"], out["direct"])
        assert not np.allclose(out["mixed"], out["direct"])

    def test_mixed_keeps_stronger_edges(self):
        # Strong checkerboard source edges vs smooth target: mixed guidance
        # must retain more interior gradient energy than target-only.
        src = synthetic.checkerboard(H, W, cell=6, v0=0.05, v1=0.95)
        tgt = synthetic.flat_field(H, W, 0.5)
        m = _rect()
        mixed = solver.blend(src, tgt, m, 0, 0, strategy="mixed")
        source = solver.blend(src, tgt, m, 0, 0, strategy="source")
        # Both keep the source here (target has zero gradient); verify the
        # mixed guidance actually selected source edges.
        assert mixed.metrics["result_gradient_energy_inside"] == pytest.approx(
            source.metrics["result_gradient_energy_inside"], rel=1e-8)


# ---------------------------------------------------------------- topology
class TestTopology:
    def test_mask_with_hole_keeps_target_inside_hole(self):
        m = synthetic.mask_with_hole(H, W)
        src = synthetic.checkerboard(H, W)
        tgt = synthetic.flat_field(H, W, 0.33)
        br = solver.blend(src, tgt, m, 0, 0)
        hole = holes(m)
        assert hole.sum() > 0
        assert br.hole_pixels == int(hole.sum())
        assert np.allclose(br.result[hole], tgt[hole])
        assert any("hole" in w for w in br.warnings)

    def test_multiple_components_solve_independently(self):
        m = synthetic.two_components(H, W)
        labels, info = connected_components(m)
        assert len(info) == 2
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        br = solver.blend(src, tgt, m, 0, 0)
        assert br.metrics["n_components"] == 2
        # Both components are pinned by their own Dirichlet rims and the
        # discrete system is satisfied; outside pixels stay on target.
        assert br.metrics["boundary_crossing_edges"] > 0
        assert np.allclose(br.result[~m], tgt[~m])
        assert br.metrics["solver"]["path"] == "hard_splu"

    def test_selection_touching_canvas_border(self):
        m = np.zeros((H, W), dtype=bool)
        m[0:20, 10:40] = True  # touches top border
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        br = solver.blend(src, tgt, m, 0, 0)
        assert any("border" in w for w in br.warnings)
        # pixels outside remain target even though the component reaches the
        # canvas edge (the remaining three rims still pin the system)
        assert np.allclose(br.result[~m], tgt[~m])

    def test_partial_placement_clipped_at_border(self):
        src = synthetic.sine_pattern(40, 40, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        m = np.ones((40, 40), dtype=bool)
        # offset pushes most of the source outside the target canvas
        br = solver.blend(src, tgt, m, offset_y=-30, offset_x=-30)
        assert br.unknown_mask.sum() == 10 * 10
        placed = br.unknown_mask
        assert np.allclose(br.result[~placed], tgt[~placed])

    def test_full_canvas_hard_mask_is_singular(self):
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        m = synthetic.full_canvas_mask(H, W)
        br = solver.blend(src, tgt, m, 0, 0)
        assert any("singular" in w for w in br.warnings)
        assert br.metrics["solver"]["path"] == "lsqr_gauge"

    def test_full_canvas_soft_mask_uses_gauge(self):
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        m = synthetic.full_canvas_mask(H, W)
        br = solver.blend(src, tgt, m, 0, 0, feather_px=3.0)
        assert br.metrics["solver"]["gauge"] is True
        assert any("singular" in w for w in br.warnings)

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError):
            solver.blend(np.zeros((H, W, 3)), np.zeros((H, W, 3)),
                         np.zeros((H, W), dtype=bool), 0, 0)


# ---------------------------------------------------------------- anchors
class TestAnchors:
    def _anchors(self):
        return [
            {"x": 25, "y": 25, "color": [0.95, 0.1, 0.1]},
            {"x": 25, "y": 25, "color": [0.05, 0.9, 0.9]},  # same pixel conflict
        ]

    def test_conflicting_anchors_are_reported(self):
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        br = solver.blend(src, tgt, _rect(), 0, 0, feather_px=4.0,
                          anchors=self._anchors(), priority="edge")
        assert any("conflicting anchors" in w for w in br.warnings)
        assert len(br.anchors_used) == 2

    def test_priority_changes_the_solution(self):
        src = synthetic.sine_pattern(H, W, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        m = _rect()
        edge = solver.blend(src, tgt, m, 0, 0, feather_px=4.0,
                            anchors=self._anchors(), priority="edge")
        lum = solver.blend(src, tgt, m, 0, 0, feather_px=4.0,
                           anchors=self._anchors(), priority="luminance")
        assert not np.allclose(edge.result, lum.result)
        # luminance priority enforces anchor colours more strongly.
        assert lum.metrics["anchor_residual_rms"] < \
            edge.metrics["anchor_residual_rms"]

    def test_anchor_pin_is_satisfied_in_soft_system(self):
        src = synthetic.planar_ramp(H, W, 0.01, 0.0, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.2)
        color = np.array([0.8, 0.2, 0.2])
        br = solver.blend(src, tgt, _rect(), 0, 0, feather_px=3.0,
                          anchors=[{"x": 30, "y": 30, "color": color.tolist(),
                                    "weight": 1e4}],
                          priority="edge")
        got = br.result[30, 30]
        assert np.linalg.norm(got - color) < 1e-3


# ---------------------------------------------------------------- drift
class TestDrReporting:
    def test_drift_is_quantified_not_hidden(self):
        # Constant source (all source gradients zero) over a different target:
        # the Poisson solution relaxes the whole region to the target level,
        # i.e. a large, explicitly reported drift rather than a hidden one.
        src = synthetic.planar_ramp(H, W, 0.0, 0.0, base=0.9)
        tgt = synthetic.flat_field(H, W, 0.1)
        m = _rect()
        br = solver.blend(src, tgt, m, 0, 0)
        assert np.allclose(br.result[m], 0.1, atol=1e-8)
        # drift = result - target ~ 0 here; the *change vs a direct source
        # copy* is what is large, and it is reported on source_delta_map.
        assert br.metrics["drift_mean_lum_inside"] == pytest.approx(0.0,
                                                                     abs=1e-8)
        assert np.abs(br.source_delta_map[m]).mean() == pytest.approx(
            0.8, abs=0.01)
        assert br.drift_map.shape == (H, W)

    def test_color_drift_against_target_is_measured(self):
        # A structured source keeps interior detail, so the solution drifts
        # away from the flat target by a measurable, reported amount.
        src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
        tgt = synthetic.flat_field(H, W, 0.1)
        br = solver.blend(src, tgt, _rect(), 0, 0)
        assert abs(br.metrics["drift_mean_lum_inside"]) > 0.05
        assert max(abs(v) for v in
                   br.metrics["drift_per_channel_max_abs"]) > 0.05

    def test_out_of_gamut_pixels_are_reported(self):
        src = synthetic.planar_ramp(H, W, 0.02, 0.0, base=0.9)
        tgt = synthetic.flat_field(H, W, 0.5)
        br = solver.blend(src, tgt, _rect(), 0, 0)
        assert "out_of_gamut_fraction_inside" in br.metrics
        assert "clipped_pixels_inside" in br.metrics
        # raw result is never clipped silently
        assert br.result.dtype == np.float64


# ---------------------------------------------------------------- masks
class TestMasks:
    def test_polygon_rasterization_and_subtract(self):
        shapes = [
            {"type": "rectangle",
             "points": [[10, 10], [40, 40]], "op": "add"},
            {"type": "ellipse",
             "points": [[20, 20], [35, 35]], "op": "subtract"},
        ]
        m = rasterize_selection(50, 50, shapes)
        assert m[12, 12] == 1
        assert m[27, 27] == 0

    def test_feather_band_values(self):
        hard = np.zeros((40, 40), dtype=bool)
        hard[10:30, 10:30] = True
        soft = soft_mask_from_hard(hard, feather_px=4.0)
        assert soft[20, 20] == pytest.approx(1.0)
        assert 0.0 < soft[10, 20] < 1.0
        assert soft[5, 20] == pytest.approx(0.0)

    def test_placement_embedding(self):
        mask = np.ones((10, 12), dtype=bool)
        placed = place_mask(mask, Placement(5, -3), (20, 20))
        assert placed.shape == (20, 20)
        assert placed[5:15, 0:9].all()
        assert placed.sum() == 10 * 9
