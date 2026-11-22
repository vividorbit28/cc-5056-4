"""Flask backend for the interactive gradient-domain fusion workbench."""
from __future__ import annotations

import base64
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, render_template, send_file

from blendtool import io_utils
from blendtool.masks import rasterize_selection, soft_mask_from_hard, Placement
from blendtool.solver import blend
from blendtool.pipeline import default_project, add_blend_step, save_project

app = Flask(__name__)
WORK = Path(__file__).resolve().parent.parent / "webwork"
WORK.mkdir(exist_ok=True)

STATE: dict = {
    "source": None,       # float HxWx3
    "target": None,
    "source_name": "",
    "target_name": "",
    "project": None,
    "last": None,         # latest BlendResult + request params
}


def _decode_image(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = io_utils.load_rgb_from_bytes(raw) if hasattr(
        io_utils, "load_rgb_from_bytes") else _load_png_bytes(raw)
    return arr


def _load_png_bytes(raw: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _encode_png(arr: np.ndarray) -> str:
    png = io_utils.to_uint8(np.clip(arr, 0.0, 1.0))
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(png), mode="RGB").save(
        buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode("ascii")


def _encode_gray_png(arr: np.ndarray) -> str:
    a8 = np.clip(np.rint(arr / max(arr.max(), 1e-12) * 255.0), 0,
                 255).astype(np.uint8)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(a8, mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode("ascii")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    payload = request.get_json(force=True)
    kind = payload["kind"]
    img = _decode_image(payload["data_url"])
    if kind == "source":
        STATE["source"] = img
        STATE["source_name"] = payload.get("name", "source")
    else:
        STATE["target"] = img
        STATE["target_name"] = payload.get("name", "target")
    if STATE["source"] is not None and STATE["target"] is not None:
        STATE["project"] = default_project(STATE["source_name"],
                                           STATE["target_name"])
    return jsonify({"ok": True,
                    "shape": list(img.shape)})


@app.post("/api/demo")
def demo():
    from blendtool import synthetic
    H = W = int(request.json.get("size", 96))
    STATE["source"] = synthetic.sine_pattern(H, W, freq=0.08, amp=0.22,
                                             base=0.65)
    STATE["target"] = synthetic.planar_ramp(H, W, 0.003, -0.002, base=0.22)
    STATE["source_name"] = "demo_source"
    STATE["target_name"] = "demo_target"
    STATE["project"] = default_project("demo_source", "demo_target")
    return jsonify({"ok": True, "source": _encode_png(STATE["source"]),
                    "target": _encode_png(STATE["target"]),
                    "shape": [H, W, 3]})


def _build_mask(params: dict):
    sh, sw = STATE["source"].shape[:2]
    hard = rasterize_selection(sw, sh, params["shapes"])
    return hard


def _placement(params: dict, hard: np.ndarray) -> Placement:
    ys, xs = np.where(hard)
    anchor = params.get("anchor_xy")
    if anchor is None:
        anchor = [float(xs.mean()), float(ys.mean())]
    tx, ty = params["target_xy"]
    return Placement(offset_y=int(round(ty - anchor[1])),
                     offset_x=int(round(tx - anchor[0])))


@app.post("/api/blend")
def run_blend():
    if STATE["source"] is None or STATE["target"] is None:
        return jsonify({"error": "upload source and target first"}), 400
    params = request.get_json(force=True)
    params.setdefault("shapes", [])
    params.setdefault("feather_px", 0.0)
    params.setdefault("strategy", "source")
    params.setdefault("priority", "edge")
    params.setdefault("target_xy", [
        STATE["target"].shape[1] // 2, STATE["target"].shape[0] // 2])
    params.setdefault("anchors", [])
    try:
        hard = _build_mask(params)
        if hard.sum() == 0:
            return jsonify({"error": "empty selection"}), 400
        placement = _placement(params, hard)
        br = blend(
            STATE["source"], STATE["target"], hard,
            placement.offset_y, placement.offset_x,
            feather_px=float(params["feather_px"]),
            strategy=params["strategy"],
            priority=params["priority"],
            anchors=params["anchors"],
            anchor_weight=float(params.get("anchor_weight", 50.0)),
            luminance_weight=float(params.get("luminance_weight", 0.25)),
            gradient_weight_scale=float(
                params.get("gradient_weight_scale", 1.0)))
    except Exception as exc:  # numerical failures reported, never hidden
        return jsonify({"error": repr(exc)}), 500

    STATE["last"] = {"result": br, "params": params}
    if STATE["project"] is not None:
        step = add_blend_step(STATE["project"],
                              name=f"{params['strategy']} blend")
        step["params"].update(params)

    drift_vmax = float(np.percentile(np.abs(br.drift_map), 99)) or 1.0
    return jsonify({
        "ok": True,
        "result": _encode_png(br.result_display),
        "residual": _encode_gray_png(br.residual_map),
        "boundary": _encode_png(
            io_utils.colorize_magnitude(br.boundary_map)),
        "drift": _encode_png(
            io_utils.colorize_diverging(br.drift_map, drift_vmax)),
        "source_delta": _encode_png(
            io_utils.colorize_diverging(br.source_delta_map,
                                        float(np.percentile(
                                            np.abs(br.source_delta_map), 99))
                                        or 1.0)),
        "anchor_residual": _encode_gray_png(br.anchor_residual_map),
        "metrics": br.metrics,
        "warnings": br.warnings,
        "components": br.components,
        "hole_pixels": br.hole_pixels,
        "anchors_used": br.anchors_used,
    })


@app.get("/api/project")
def get_project():
    if STATE["project"] is None:
        return jsonify({"project": None})
    return jsonify({"project": STATE["project"]})


@app.post("/api/project/save")
def save():
    if STATE["project"] is None:
        return jsonify({"error": "no project"}), 400
    name = f"project_{uuid.uuid4().hex[:8]}.json"
    path = WORK / name
    save_project(STATE["project"], str(path))
    return send_file(path, as_attachment=True, download_name=name)


@app.post("/api/project/load")
def load():
    from blendtool.pipeline import load_project
    f = request.files["project"]
    try:
        project = json.loads(f.read().decode("utf-8"))
        load_project_from_dict = project  # validated below
        if project.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
    except Exception as exc:
        return jsonify({"error": repr(exc)}), 400
    STATE["project"] = project
    return jsonify({"ok": True, "project": project})


@app.post("/api/export")
def export():
    if STATE["last"] is None:
        return jsonify({"error": "nothing to export"}), 400
    kind = request.json.get("kind", "result")
    br = STATE["last"]["result"]
    choices = {
        "result": br.result_display,
        "residual": np.repeat(
            (br.residual_map / max(br.residual_map.max(), 1e-12))[..., None],
            3, axis=2),
        "drift": io_utils.colorize_diverging(br.drift_map),
    }
    arr = choices.get(kind, br.result_display)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(io_utils.to_uint8(arr), mode="RGB").save(
        buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png",
                     download_name=f"{kind}.png", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
