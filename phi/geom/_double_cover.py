from dataclasses import dataclass
from functools import cached_property

from phi.geom._spline_sheet import BSplineSheet
from phiml import Tensor, spatial, ZERO_GRADIENT, vec, wrap, dual, stack
from phiml.dataclasses import sliceable
from phiml.math import pad, spatial_gradient, scatter
from phiml.math._ops import pad_to_uniform
from ._functions import cross, normalize


@sliceable(keepdims='vector')
@dataclass(frozen=True)
class DoubleCover:

    points: Tensor
    """ (N:s, N:s, vector)"""
    radius: Tensor
    """2D spatial tensor matching vertices"""
    degree: int
    """ Spline degree. """
    crease: Tensor
    """ (N:s, uv=2) """
    variable_attrs = ('points', 'radius', 'crease')

    def __post_init__(self):
        assert 'vector' in self.points.shape
        assert spatial(self.points) in self.radius.shape, f"radius must be per-point but got shape {self.radius.shape}"

    @cached_property
    def surface(self):
        front_back = wrap([1, -1], dual(side='front,back'))
        is_back = wrap([False, True], front_back.shape)
        # --- Compute front/back spline control points, add outer layer of centered control points where front and back join ---
        radius = pad(self.radius, {spatial: (1, 1)}, 0)
        center_points = pad(self.points, {spatial: (1, 1)}, ZERO_GRADIENT)
        deltas = spatial_gradient(self.points, padding=ZERO_GRADIENT)
        normals = normalize(cross(deltas.gradient['u'], deltas.gradient['v']))
        normals = pad(normals, {spatial: (1, 1)}, ZERO_GRADIENT)
        points = center_points + normals * radius * front_back
        # --- Pad tensors to max point count ---
        size = max(wrap(points.shape.get_size('u')).max, wrap(points.shape.get_size('v')).max)
        points = pad_to_uniform(points, spatial(u=size, v=size))
        crease = pad_to_uniform(self.crease)
        # --- Fix multiplicity at corners ---
        cu, cv = self.points.shape.get_size('u') + 1, self.points.shape.get_size('v') + 1
        corners = [
            [vec(u=0, v=0), [vec(u=0, v=2), vec(u=2, v=1)], [vec(u=2, v=0), vec(u=1, v=2)]],
            [vec(u=cu, v=0), [vec(u=cu, v=2), vec(u=cu-2, v=1)], [vec(u=cu-2, v=0), vec(u=cu-1, v=2)]],
            [vec(u=0, v=cv), [vec(u=0, v=cv-2), vec(u=2, v=cv-1)], [vec(u=2, v=cv), vec(u=1, v=cv-2)]],
            [vec(u=cu, v=cv), [vec(u=cu, v=cv-2), vec(u=cu-2, v=cv-1)], [vec(u=cu-2, v=cv), vec(u=cu-1, v=cv-2)]],
        ]
        corner = stack([c for c, *_ in corners], 'corners:i')
        u1 = stack([u for _, (u, _), _ in corners], 'corners:i')
        u2 = stack([u for _, (_, u), _ in corners], 'corners:i')
        v1 = stack([v for _, _, (v, _) in corners], 'corners:i')
        v2 = stack([v for _, _, (_, v) in corners], 'corners:i')
        points_u = scatter(points, corner, 1e-3 * (points[u1] - points[u2]), 'add')
        points_v = scatter(points, corner, 1e-3 * (points[v1] - points[v2]), 'add')
        # points_u = points_v = points
        # for corner, (u1, u2), (v1, v2) in corners:
        #     points_u = scatter(points_u, corner, 1e-3 * (points[u1] - points[u2]), 'add')
        #     points_v = scatter(points_v, corner, 1e-3 * (points[v1] - points[v2]), 'add')
        derivative_points = {'u': points_u, 'v': points_v}
        # --- Construct splines ---
        res = vec(u=self.points.shape.get_size('u') + 2, v=self.points.shape.get_size('v') + 2)
        return BSplineSheet(points, self.degree, crease, res, flip_normals=is_back, derivative_points=derivative_points, sample_grid_margin=.01)
