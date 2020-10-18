from __future__ import annotations

from phi import math
from phi.math import extrapolation


class Material:
    """
    Defines the extrapolation modes / boundary conditions for a surface.
    The surface can be an obstacle or the domain boundary.
    """
    def __init__(self, name, grid_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation):
        self.name = name
        self.grid_extrapolation = grid_extrapolation
        self.vector_extrapolation = vector_extrapolation
        self.active_extrapolation = active_extrapolation
        self.accessible_extrapolation = accessible_extrapolation

    def __repr__(self):
        return self.name

    @staticmethod
    def as_material(obj: Material or tuple or list or dict) -> Material:
        if isinstance(obj, Material):
            return obj
        if isinstance(obj, (tuple, list)):
            axes = [math.GLOBAL_AXIS_ORDER.axis_name(i, len(obj)) for i in range(len(obj))]
            obj = {ax: mat for ax, mat in zip(axes, obj)}
        if isinstance(obj, dict):
            grid_extrapolation = {ax: mat.grid_extrapolation for ax, mat in obj.items()}
            vector_extrapolation = {ax: mat.vector_extrapolation for ax, mat in obj.items()}
            active_extrapolation = {ax: mat.active_extrapolation for ax, mat in obj.items()}
            accessible_extrapolation = {ax: mat.accessible_extrapolation for ax, mat in obj.items()}
            return Material('mixed', grid_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation)
        raise NotImplementedError()


OPEN = Material('open', extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ONE)
CLOSED = NO_STICK = SLIPPERY = Material('slippery', extrapolation.BOUNDARY, extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO)
NO_SLIP = STICKY = Material('sticky', extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO)
PERIODIC = Material('periodic', extrapolation.PERIODIC, extrapolation.PERIODIC, extrapolation.ONE, extrapolation.ONE)
