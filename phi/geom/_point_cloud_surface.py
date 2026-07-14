from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Tuple

from phiml import math, Tensor, instance, imean, where
from phiml.dataclasses import sliceable
from phiml.math import find_closest, normalize

from ._geom import Geometry


@sliceable(keepdims='vector')
@dataclass(frozen=True)
class PointCloudSurface(Geometry):
    points: Tensor
    k: int

    def __post_init__(self):
        assert 'vector' in self.points.shape, f"PointCloudSurface requires 'vector' dim but got {type(self.points)} with shape {self.points.shape}."
        assert instance(self.points), f"points must have an instance dim but got shape {self.points.shape}."
        assert self.k > 0, f"PointCloudSurface requires k > 0 but got k={self.k}."
        assert self.k < instance(self.points).volume, f"PointCloudSurface requires k < num points, but got k={self.k} for {instance(self.points).volume} points."

    @cached_property
    def shape(self):
        return self.points.shape

    @cached_property
    def support_at(self) -> Callable[[Tensor], Tensor]:
        return find_closest(self.points, list_dim=instance(k=self.k))

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        closest = self.points[self.support_at(location)]
        center = imean(closest)
        centered = closest - center
        cov = math.sum(centered * centered.vector.T, 'k')
        _, eigvals, eigvecs = math.eig(cov)
        normal = math.min(eigvecs, '~eigenvalues', key=eigvals.eigenvalues.T)
        distance = normal.vector @ (location - center)
        return abs(distance), -distance * normal, where(distance < 0, -normal, normal), None, None

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return self.approximate_closest_surface(location)[0]

    @cached_property
    def normals(self):
        return self.approximate_closest_surface(self.points)[2]
