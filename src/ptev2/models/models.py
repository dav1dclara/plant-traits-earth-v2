"""Compatibility shim for legacy checkpoint/config targets.

Older runs reference `ptev2.models.models.*` in serialized configs.
This module re-exports the current implementations from `multitask`
so old checkpoints can be loaded in the unified main pipeline.
"""

from ptev2.models.multitask import (  # noqa: F401
    GatedMMoEModel,
    GatedMMoEModelV3,
    MTLModel,
    STLModel,
)
