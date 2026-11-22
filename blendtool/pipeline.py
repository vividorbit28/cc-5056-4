"""Non-destructive, parameter-saveable processing chain.

A project is a plain JSON document referencing image files by path and
holding an ordered list of steps. Re-running the chain always starts from
the original images; intermediate results are cached but never overwrite
the sources. Blend steps carry every numerical parameter (mask shapes,
feather, guidance strategy, priority, anchors) so results are reproducible.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import io_utils
from .masks import rasterize_selection, soft_mask_from_hard, Placement
from .solver import blend, BlendResult

SCHEMA_VERSION = 1


def default_project(source_path: str = "", target_path: str = "") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created": time.time(),
        "source_path": source_path,
        "target_path": target_path,
        "steps": [],
    }


def add_blend_step(project: dict, *, name: str = "blend") -> dict:
    step = {
        "id": f"step-{len(project['steps']) + 1}-{int(time.time() * 1000)}",
        "name": name,
        "enabled": True,
        "op": "blend",
        "params": {
            "shapes": [],              # source-image coordinates
            "feather_px": 0.0,
            "strategy": "source",      # source | mixed | direct
            "priority": "edge",        # edge | luminance
            "target_xy": [0, 0],       # placement point on target
            "anchor_xy": None,         # source point mapped to target_xy
            "anchors": [],             # target-coord colour anchors
            "anchor_weight": 50.0,
            "luminance_weight": 0.25,
            "gradient_weight_scale": 1.0,
        },
    }
    project["steps"].append(step)
    return step


def save_project(project: dict, path: str) -> None:
    Path(path).write_text(json.dumps(project, indent=2, ensure_ascii=False))


def load_project(path: str) -> dict:
    project = json.loads(Path(path).read_text())
    if project.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported project schema version: "
            f"{project.get('schema_version')} (expected {SCHEMA_VERSION})")
    return project


@dataclass
class StepOutput:
    step_id: str
    result: BlendResult | None
    image: np.ndarray
    metrics: dict
    warnings: list[str]
    error: str | None = None


class Pipeline:
    """Runs a project's steps against fixed source/target arrays."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source
        self.target = target
        self.outputs: list[StepOutput] = []

    @classmethod
    def from_project_paths(cls, project: dict) -> "Pipeline":
        source = io_utils.load_rgb(project["source_path"])
        target = io_utils.load_rgb(project["target_path"])
        return cls(source, target)

    def run(self, project: dict) -> list[StepOutput]:
        self.outputs = []
        canvas = self.target.copy()
        for step in project["steps"]:
            if not step.get("enabled", True):
                continue
            if step["op"] != "blend":
                raise ValueError(f"unknown op: {step['op']}")
            try:
                br = self._run_blend(step["params"], canvas)
                canvas = br.result_display.copy()
                self.outputs.append(StepOutput(
                    step_id=step["id"], result=br, image=canvas,
                    metrics=copy.deepcopy(br.metrics),
                    warnings=list(br.warnings)))
            except Exception as exc:  # reported, not silently dropped
                self.outputs.append(StepOutput(
                    step_id=step["id"], result=None, image=canvas,
                    metrics={}, warnings=[], error=repr(exc)))
                raise
        return self.outputs

    def _run_blend(self, p: dict, current_target: np.ndarray) -> BlendResult:
        sh, sw = self.source.shape[:2]
        hard = rasterize_selection(sw, sh, p["shapes"])
        if hard.sum() == 0:
            raise ValueError("step has an empty selection")
        anchor = p.get("anchor_xy")
        if anchor is None:
            ys, xs = np.where(hard)
            anchor = [float(xs.mean()), float(ys.mean())]
        placement = Placement(
            offset_y=int(round(float(p["target_xy"][1]) - float(anchor[1]))),
            offset_x=int(round(float(p["target_xy"][0]) - float(anchor[0]))))
        return blend(
            self.source, current_target, hard,
            placement.offset_y, placement.offset_x,
            feather_px=float(p.get("feather_px", 0.0)),
            strategy=p.get("strategy", "source"),
            priority=p.get("priority", "edge"),
            anchors=p.get("anchors", []),
            anchor_weight=float(p.get("anchor_weight", 50.0)),
            luminance_weight=float(p.get("luminance_weight", 0.25)),
            gradient_weight_scale=float(p.get("gradient_weight_scale", 1.0)))


def export_result(image: np.ndarray, path: str) -> None:
    io_utils.save_rgb(path, image)
