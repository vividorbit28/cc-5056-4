"""Gradient-domain local image fusion tool.

Modules:
  masks    - soft/hard selection masks, connected components, placement
  gradient - guidance gradient fields (source / mixed / direct)
  solver   - builds and solves the discrete gradient-domain linear systems
  metrics  - residuals, boundary continuity, drift, detail statistics
  synthetic- generators with analytically known gradients for verification
  pipeline - non-destructive, parameter-saveable processing chain
  io_utils - image IO helpers (float [0,1] RGB arrays)
"""

__all__ = [
    "masks",
    "gradient",
    "solver",
    "metrics",
    "synthetic",
    "pipeline",
    "io_utils",
]
