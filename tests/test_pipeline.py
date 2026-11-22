"""Non-destructive pipeline: parameter persistence and reproducibility."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blendtool import synthetic, io_utils
from blendtool.pipeline import (Pipeline, default_project, add_blend_step,
                                save_project, load_project, export_result)


def _images(tmp_path: Path):
    H = W = 48
    src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
    tgt = synthetic.flat_field(H, W, 0.2)
    sp = tmp_path / "src.png"
    tp = tmp_path / "tgt.png"
    io_utils.save_rgb(str(sp), src)
    io_utils.save_rgb(str(tp), tgt)
    return src, tgt, sp, tp


def test_project_roundtrip_and_reproducibility(tmp_path):
    src, tgt, sp, tp = _images(tmp_path)
    project = default_project(str(sp), str(tp))
    step = add_blend_step(project)
    step["params"]["shapes"] = [
        {"type": "rectangle", "op": "add",
         "points": [[10, 10], [38, 38]]}]
    step["params"]["strategy"] = "source"
    step["params"]["feather_px"] = 4.0
    step["params"]["target_xy"] = [24, 24]
    pp = tmp_path / "project.json"
    save_project(project, str(pp))

    reloaded = load_project(str(pp))
    assert reloaded["steps"][0]["params"] == project["steps"][0]["params"]

    out1 = Pipeline(src, tgt).run(project)
    out2 = Pipeline(src, tgt).run(reloaded)
    assert len(out1) == len(out2) == 1
    assert np.array_equal(out1[0].image, out2[0].image)
    assert out1[0].metrics["mode"] == "soft_wls"


def test_pipeline_does_not_modify_inputs(tmp_path):
    src, tgt, sp, tp = _images(tmp_path)
    src_before = src.copy()
    tgt_before = tgt.copy()
    project = default_project(str(sp), str(tp))
    step = add_blend_step(project)
    step["params"]["shapes"] = [
        {"type": "rectangle", "op": "add",
         "points": [[5, 5], [40, 40]]}]
    step["params"]["strategy"] = "mixed"
    Pipeline(src, tgt).run(project)
    assert np.array_equal(src, src_before)
    assert np.array_equal(tgt, tgt_before)


def test_disabled_and_multi_chain_steps(tmp_path):
    src, tgt, sp, tp = _images(tmp_path)
    project = default_project(str(sp), str(tp))
    s1 = add_blend_step(project, name="first")
    s1["params"]["shapes"] = [
        {"type": "rectangle", "op": "add",
         "points": [[8, 8], [25, 25]]}]
    s2 = add_blend_step(project, name="second-disabled")
    s2["enabled"] = False
    s2["params"]["shapes"] = [
        {"type": "rectangle", "op": "add",
         "points": [[20, 20], [40, 40]]}]
    outs = Pipeline(src, tgt).run(project)
    assert len(outs) == 1
    assert outs[0].step_id == s1["id"]


def test_empty_selection_is_reported(tmp_path):
    src, tgt, sp, tp = _images(tmp_path)
    project = default_project(str(sp), str(tp))
    s = add_blend_step(project)
    s["params"]["shapes"] = []
    with pytest.raises(ValueError):
        Pipeline(src, tgt).run(project)


def test_export_writes_image(tmp_path):
    img = synthetic.checkerboard(16, 16)
    out = tmp_path / "result.png"
    export_result(img, str(out))
    assert out.exists()
    loaded = io_utils.load_rgb(str(out))
    assert loaded.shape == img.shape


def test_schema_version_guard(tmp_path):
    bad = {"schema_version": 999, "source_path": "", "target_path": "",
           "steps": []}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_project(str(p))
