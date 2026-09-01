"""Baseline experiments vendored from the deepracer-genesis ``examples/`` package.

Only what the studies here subclass is copied; see camera.py for why.
"""

from experiments.baselines.camera import (
    CameraCpu,
    CameraMadronaDr,
    CameraMaxDr,
    CameraNyx,
    CameraZoo,
)

__all__ = [
    "CameraCpu",
    "CameraMadronaDr",
    "CameraMaxDr",
    "CameraNyx",
    "CameraZoo",
]
