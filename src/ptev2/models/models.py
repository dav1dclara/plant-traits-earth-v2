"""Compatibility shim for legacy checkpoint/config targets.

Older runs reference `ptev2.models.models.*` in serialized configs.
This module re-exports the current implementations from `multitask`
so old checkpoints can be loaded in the unified main pipeline.
"""

from ptev2.models.mmoe import GatedMMoEModel, GatedMMoEModelV3  # noqa: F401
from ptev2.models.mtl import MTLModel  # noqa: F401
from ptev2.models.pixel_mlp import PixelMLP  # noqa: F401
from ptev2.models.stl import STLModel  # noqa: F401
