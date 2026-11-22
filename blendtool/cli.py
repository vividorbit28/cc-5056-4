"""Command-line interface for headless blending and verification.

Examples
--------
Run a saved project and export each step's result plus residual maps:

    python -m blendtool.cli run project.json --outdir out/

Blend a rectangular region directly:

    python -m blendtool.cli blend source.png target.png \\
        --rect 10,10,80,80 --target 120,120 --strategy source \\
        --feather 4 --out result.png --report report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import io_utils
from .masks import rasterize_selection
from .pipeline import (Pipeline, default_project, add_blend_step,
                       save_project, load_project)
from .solver import blend


def _write_maps(br, outdir: Path, tag: str):
    io_utils.save_rgb(str(outdir / f"{tag}_result.png"), br.result_display)
    io_utils.save_gray(str(outdir / f"{tag}_residual.png"),
                       np.clip(br.residual_map /
                               max(br.residual_map.max(), 1e-12), 0, 1))
    io_utils.save_rgb(str(outdir / f"{tag}_drift.png"),
                      np.clip(io_utils.colorize_diverging(br.drift_map), 0, 1))
    io_utils.save_rgb(str(outdir / f"{tag}_boundary.png"),
                      np.clip(io_utils.colorize_magnitude(br.boundary_map),
                              0, 1))


def cmd_run(args):
    project = load_project(args.project)
    pipe = Pipeline.from_project_paths(project)
    outputs = pipe.run(project)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    for k, out in enumerate(outputs):
        if out.result is not None:
            tag = f"step{k+1}"
            _write_maps(out.result, outdir, tag)
        report.append({"step_id": out.step_id, "metrics": out.metrics,
                       "warnings": out.warnings, "error": out.error})
    (outdir / "report.json").write_text(json.dumps(report, indent=2,
                                                    ensure_ascii=False,
                                                    default=str))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def cmd_blend(args):
    source = io_utils.load_rgb(args.source)
    target = io_utils.load_rgb(args.target)
    sh, sw = source.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in args.rect.split(","))
    shapes = [{"type": "rectangle", "op": "add",
               "points": [[x0, y0], [x1, y1]]}]
    hard = rasterize_selection(sw, sh, shapes)
    tx, ty = (int(v) for v in args.target_xy.split(","))
    anchors = json.loads(args.anchors) if args.anchors else None
    br = blend(source, target, hard,
               offset_y=ty - (y0 + y1) // 2,
               offset_x=tx - (x0 + x1) // 2,
               feather_px=args.feather, strategy=args.strategy,
               priority=args.priority, anchors=anchors,
               seam_column=args.seam)
    io_utils.save_rgb(args.out, br.result_display)
    report = {"metrics": br.metrics, "warnings": br.warnings,
              "components": br.components, "anchors": br.anchors_used}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2,
                                                 ensure_ascii=False,
                                                 default=str))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def cmd_demo(args):
    """Generate synthetic images with known gradients and verify the solver."""
    from . import synthetic
    H = W = args.size
    src = synthetic.sine_pattern(H, W, freq=0.1, amp=0.2, base=0.7)
    tgt = synthetic.planar_ramp(H, W, 0.002, -0.001, base=0.25)
    mask = synthetic.rect_mask(H, W, H // 5, H // 5,
                               4 * H // 5, 4 * W // 5)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    io_utils.save_rgb(str(outdir / "source.png"), src)
    io_utils.save_rgb(str(outdir / "target.png"), tgt)
    summary = {}
    for strategy in ("source", "mixed", "direct"):
        br = blend(src, tgt, mask, 0, 0, strategy=strategy,
                   feather_px=args.feather if strategy != "direct" else 0.0)
        tag = strategy
        io_utils.save_rgb(str(outdir / f"{tag}.png"), br.result_display)
        summary[tag] = br.metrics
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a saved project pipeline")
    pr.add_argument("project")
    pr.add_argument("--outdir", required=True)
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("blend", help="single rectangular blend")
    pb.add_argument("source")
    pb.add_argument("target")
    pb.add_argument("--rect", required=True, help="x0,y0,x1,y1 in source")
    pb.add_argument("--target", dest="target_xy", required=True,
                    help="x,y placement point on target")
    pb.add_argument("--feather", type=float, default=0.0)
    pb.add_argument("--strategy", choices=["source", "mixed", "direct"],
                    default="source")
    pb.add_argument("--priority", choices=["edge", "luminance"],
                    default="edge")
    pb.add_argument("--anchors", default=None,
                    help='JSON list of anchor dicts')
    pb.add_argument("--seam", type=int, default=None,
                    help="target-canvas x of a vertical seam partition")
    pb.add_argument("--out", required=True)
    pb.add_argument("--report", default=None)
    pb.set_defaults(func=cmd_blend)

    pd = sub.add_parser("demo", help="synthetic verification demo")
    pd.add_argument("--size", type=int, default=64)
    pd.add_argument("--feather", type=float, default=5.0)
    pd.add_argument("--outdir", default="demo_out")
    pd.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
