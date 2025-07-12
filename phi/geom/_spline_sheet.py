from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Tuple, Dict

from phiml import math, Tensor, spatial, Shape, vec, linspace, dual, stack, channel, wrap, instance, clip, batch
from phiml.dataclasses import sliceable
from phiml.math import stop_gradient, copy_with
from ._functions import solve2x2, vec_length, cross, normalize
from ._geom import Geometry
from ._mesh import Mesh
from ._mesh_builder import MeshBuilder
from ._spline import b_spline_knots, eval_nurbs_bases


@sliceable(keepdims='vector')
@dataclass(frozen=True)
class BSplineSheet(Geometry):
    """
    2D B-spline manifold. Defined by n×m control points and n+m crease values.
    Multiple surfaces can be combined into a single object by listing them along dual dims.
    This is different from stacking along instance dims, which represents different objects.
    Concretely, the SDF of a point is the lowest across all instanced objects, but the nearest across (dual) surfaces.

    For better parallelization, the points tensor must be padded to N×N where `N=max(n,m)`. The number of used points is stored in `res`.
    """

    points: Tensor
    """ Control point locations of shape (max_res, max_res, vector). """
    degree: int
    """ Spline degree (1=linear, 2=quadratic, 3=cubic). """
    crease: Tensor
    """ Crease per non-end row of shape (res-2, uv=2) """
    res: Tensor
    """ Actual spline resolution along u and v as vector (res_u, res_v). Points >= res are ignored. """
    flip_normals: Union[bool, Tensor]
    """ Whether the face normals point the other way. Flips outside/inside. """
    derivative_points: Dict[str, Tensor] = field(default_factory= lambda: {})
    """ Use these points instead of `points` for derivative computations along `u` or `v`. """
    # --- Sampled surface points used for visualization and initial UV guess ---
    sample_res_u: int = 16
    sample_res_v: int = 16
    sample_grid_margin: float = 1e-3
    # --- PhiML meta-info ---
    variable_attrs = ('points', 'crease')

    def __post_init__(self):
        assert set(spatial(self.points).names) == {'u', 'v'}, f"points must have spatial dims 'u' and 'v', but got {self.points.shape}"
        assert len(set(spatial(self.points).sizes)) == 1, f"points must have the same size in u and v, but got {self.points.shape}"

    @cached_property
    def shape(self):
        return self.points.shape

    @property
    def resolution(self):
        return spatial(u=self.res['u'], v=self.res['u'])

    @cached_property
    def knots(self):
        n_alloc = spatial(self.points).get_size('u')
        def single_knots(res: int, crease):
            return b_spline_knots(spatial(ctrl_pt=res), self.degree, crease=crease, pad_to=n_alloc)
        return math.map(single_knots, self.res, self.crease, dims=self.res.shape & batch(self.crease))

    def eval_pos(self, uv: Tensor, selection: Tensor = None):
        bases_u, bases_v = eval_nurbs_bases(uv, self.knots, weights=None).vector
        bases = bases_u.ctrl_pt.as_dual('u') * bases_v.ctrl_pt.as_dual('v')
        return bases @ self.points[selection]

    def eval_surf(self, uv: Tensor, selection: Tensor = None):
        bases_u, bases_v = eval_nurbs_bases(uv, self.knots, weights=None, compute_derivative=True).vector
        bases_u, du = bases_u.ctrl_pt.as_dual('u').d_order
        bases_v, dv = bases_v.ctrl_pt.as_dual('v').d_order
        sel_points = self.points[selection]
        pos = (bases_u * bases_v) @ sel_points
        tan_u = (du * bases_v) @ (self.derivative_points['u'][selection] if 'u' in self.derivative_points else sel_points)
        tan_v = (bases_u * dv) @ (self.derivative_points['v'][selection] if 'v' in self.derivative_points else sel_points)
        normal = normalize(cross(tan_u, tan_v) * math.where(math.slice(self.flip_normals, selection), -1, 1), epsilon=None)
        return pos, stack([tan_u, tan_v], dual('tangents')), normal

    @cached_property
    def _sampled_grid(self):
        m = self.sample_grid_margin
        uv_grid = vec(u=linspace(m, 1-m, spatial(u=self.sample_res_u)), v=linspace(m, 1-m, spatial(v=self.sample_res_v)))
        return (uv_grid, *self.eval_surf(uv_grid))

    def build_mesh(self, res_u: int = 32, res_v: int = 32, element_dim=instance('elements')) -> Mesh:
        uv_grid = vec(u=linspace(0, 1, spatial(u=res_u)), v=linspace(0, 1, spatial(v=res_v)))
        pos = self.eval_pos(uv_grid)
        mb = MeshBuilder(2, self.shape.non_spatial.non_channel.non_dual)
        mb.new_quads("surface", pos, wrap(0))
        return mb.build_mesh(element_dim)

    def closest_surface(self, location: Tensor, niter=3):
        params, points, tan, normal = self._sampled_grid
        closest_idx = math.find_closest(points, location)
        component_idx = closest_idx[dual(self)] if dual(self) else None
        uv = params[closest_idx]
        tan = tan[closest_idx]
        pos = points[closest_idx]
        normal = None if niter > 0 else normal[closest_idx]
        limit = .5 / 16 if niter > 0 else None  # half dt of sampled grid
        for i in range(niter):
            proj = tan.vector @ (location - pos)
            tu, tv = tan.tangents.dual
            tan_dot = tu.vector @ tv.vector
            uu, vv = (tan.vector @ tan.vector).tangents.dual
            du, dv = solve2x2(uu, tan_dot, tan_dot, vv, *proj.tangents.dual)
            duv = stack([du, dv], channel(uv))
            duv = math.nan_to_0(duv)
            step_size = .8**i
            uv += clip(duv, -limit, limit) * step_size  # we could clip this to avoid divergence
            uv = clip(uv)
            uv = stop_gradient(uv)  # we don't want to back-prop through finding the parameter. However, pos, tan, normal should be differentiable.
            pos, tan, normal = self.eval_surf(uv, selection=component_idx)
        return uv, component_idx, pos, tan, normal

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        uv, component_idx, pos, tan, normal = self.closest_surface(location)
        delta = pos - location
        dist = vec_length(delta)
        sgn_dist = math.where(normal.vector @ delta.vector < 0, dist, -dist)
        sgn_dist, delta, normal, uv = math.min((sgn_dist, delta, normal, uv), instance(self), key=sgn_dist)  # since spline sheets are not closed, we cannot take min(sgn_dist)
        return sgn_dist, delta, normal, None, uv

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        _, _, pos, _, normal = self.closest_surface(location)
        delta = pos - location
        dist = vec_length(delta)
        sgn_dist = math.where(normal.vector @ delta.vector < 0, dist, -dist)
        sgn_dist = math.min(sgn_dist, instance(self), key=sgn_dist)
        return sgn_dist

    @property
    def center(self) -> Tensor:
        return self.points

    @property
    def volume(self) -> Tensor:
        return wrap(0)
    #
    # @property
    # def faces(self) -> 'Geometry':
    #     raise NotImplementedError
    #
    # @property
    # def face_centers(self) -> Tensor:
    #     raise NotImplementedError
    #
    # @property
    # def face_areas(self) -> Tensor:
    #     raise NotImplementedError
    #
    # @property
    # def face_normals(self) -> Tensor:
    #     raise NotImplementedError
    #
    # @property
    # def boundary_elements(self) -> Dict[str, Dict[str, slice]]:
    #     raise NotImplementedError
    #
    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return dual(spline_faces=1)  # not currently used, but needs to be implemented so that Field values are not assumed to be staggered.
    #
    # @property
    # def sets(self) -> Dict[str, Shape]:
    #     return super().sets
    #
    # def get_points(self, set_key: str) -> Tensor:
    #     return super().get_points(set_key)
    #
    # def get_boundary(self, set_key: str) -> Dict[str, Dict[str, slice]]:
    #     return super().get_boundary(set_key)
    #
    # @property
    # def corners(self) -> Tensor:
    #     raise NotImplementedError
    #
    # def integrate_surface(self, face_values: Tensor, divide_volume=False) -> Tensor:
    #     return super().integrate_surface(face_values, divide_volume)
    #
    # def integrate_flux(self, flux: Tensor, divide_volume=False) -> Tensor:
    #     return super().integrate_flux(flux, divide_volume)
    #
    # def unstack(self, dimension: str) -> tuple:
    #     return super().unstack(dimension)
    #
    # def lies_inside(self, location: Tensor) -> Tensor:
    #     raise NotImplementedError
    #
    # def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
    #     return super().push(positions, outward, shift_amount)
    #
    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        # does not yet take vel(t) or segment length into account, so points are not truly uniform on the surface.
        return self.eval_pos(math.rand(channel(vector='u,v'), *shape))

    def bounding_radius(self) -> Tensor:
        return wrap(0)

    def bounding_half_extent(self) -> Tensor:
        return wrap(0)

    def at(self, center: Tensor) -> 'Geometry':
        return copy_with(self, points=center)
    #
    # def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
    #     raise NotImplementedError
    #
    # def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
    #     raise NotImplementedError
    #
    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        for v in values[1:]:
            assert v.shape.spatial == values[0].shape.spatial
        return super().__stack__(values, dim, **kwargs)
