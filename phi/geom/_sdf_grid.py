from numbers import Number
from typing import Union, Tuple, Dict, Any, Optional, Sequence

from phiml import math
from phiml.math import Shape, Tensor, spatial, channel, non_spatial, expand, instance, dual, clip, wrap
from phiml.math.magic import slicing_dict
from ._geom import Geometry
from ._functions import clip_length
from ._grid import UniformGrid
from ._box import BaseBox, Cuboid


class SDFGrid(Geometry):
    """
    Grid-based signed distance field.
    """
    def __init__(self,
                 sdf: Tensor,
                 bounds: BaseBox,
                 approximate_outside=True,
                 gradient: Tensor = None,
                 to_surface: Tensor = None, surf_normal: Tensor = None, surf_index: Tensor = None,
                 center: Tensor = None, volume: Tensor = None, bounding_radius: Tensor = None):
        """
        Args:
            sdf: Signed distance values. `Tensor` with spatial dimensions corresponding to the physical space.
                Each value samples the SDF value at the center of a virtual cell.
            bounds: Grid limits. The bounds fully enclose all virtual cells.
            approximate_outside: Whether queries outside the SDF grid should return approximate values. This requires additional computations.
            gradient: (Optional) Pre-computed gradient grid. Will be computed otherwise.
            center: (Optional) Geometry center point. Will be computed otherwise.
            volume: (Optional) Geometry volume. Will be computed otherwise.
            bounding_radius: (Optional) Geometry bounding radius around center. Will be computed otherwise.
        """
        super().__init__()
        self._sdf = sdf
        self._bounds = bounds
        self._approximate_outside = approximate_outside
        dx = bounds.size / spatial(sdf)
        if gradient is True:
            grad = math.spatial_gradient(sdf, dx=dx, difference='forward', padding=math.extrapolation.ZERO_GRADIENT, stack_dim=channel('vector'))
            self._grad = grad[{dim: slice(0, -1) for dim in spatial(sdf).names}]
        else:
            self._grad = gradient
        self._to_surface = to_surface
        self._surf_normal = surf_normal
        self._surf_index = surf_index
        if center is not None:
            self._center = center
        else:
            min_index = math.argmin(self._sdf, spatial, index_dim=channel('vector'))
            self._center = bounds.local_to_global(min_index / spatial(sdf))
        if volume is not None:
            self._volume = volume
        else:
            filled = math.sum(sdf < 0)
            self._volume = filled * math.prod(dx)
        if bounding_radius is not None:
            self._bounding_radius = bounding_radius
        else:
            points = UniformGrid(spatial(sdf), self._bounds).center
            dist = math.vec_length(points - self._center)
            dist = math.where(self._sdf <= 0, dist, 0)
            self._bounding_radius = math.max(dist)

    def __variable_attrs__(self):
        return '_sdf', '_bounds', '_grad', '_to_surface', '_surf_normal', '_surf_index', '_center', '_volume', '_bounding_radius'

    def __value_attrs__(self):
        return '_sdf',

    @property
    def values(self):
        """Signed distance grid."""
        return self._sdf

    def with_values(self, values: Tensor):
        values = expand(values, spatial(self._sdf) - spatial(values))
        return SDFGrid(values, self._bounds, self._approximate_outside, self._grad, self._center, self._volume, self._bounding_radius)

    @property
    def bounds(self) -> BaseBox:
        return self._bounds

    @property
    def size(self) -> Tensor:
        return self._bounds.size

    @property
    def resolution(self) -> Shape:
        return spatial(self._sdf)

    @property
    def dx(self) -> Tensor:
        return self._bounds.size / spatial(self._sdf)

    @property
    def points(self) -> Tensor:
        return UniformGrid(spatial(self._sdf), self._bounds).center

    @property
    def grid(self) -> UniformGrid:
        return UniformGrid(spatial(self._sdf), self._bounds)

    @property
    def center(self) -> Tensor:
        return self._center

    @property
    def shape(self) -> Shape:
        return non_spatial(self._sdf) & channel(vector=spatial(self._sdf))

    @property
    def volume(self) -> Tensor:
        return self._volume

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return math.EMPTY_SHAPE

    @property
    def corners(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support corners")

    def lies_inside(self, location: Tensor) -> Tensor:
        float_idx = (location - self._bounds.lower) / self.size * self.resolution
        sdf_val = math.grid_sample(self._sdf, float_idx - .5, math.extrapolation.ZERO_GRADIENT)
        if self._approximate_outside:
            within_bounds = self._bounds.lies_inside(location)
            return within_bounds & (sdf_val <= 0)
        else:
            return sdf_val <= 0

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._approximate_outside:
            location = self._bounds.push(location, outward=False)
        float_idx = (location - self._bounds.lower) / self.size * self.resolution
        sgn_dist = math.grid_sample(self._sdf, float_idx - .5, math.extrapolation.ZERO_GRADIENT)
        if self._to_surface is not None:
            to_surf = math.grid_sample(self._to_surface, float_idx - .5, math.extrapolation.ZERO_GRADIENT)
        else:
            sdf_grad = math.grid_sample(self._grad, float_idx - 1, math.extrapolation.ZERO_GRADIENT)
            sdf_grad = math.vec_normalize(sdf_grad)  # theoretically not necessary
            to_surf = sgn_dist * -sdf_grad
        surface_pos = location + to_surf
        if self._surf_normal is not None:
            normal = math.grid_sample(self._surf_normal, float_idx - .5, math.extrapolation.ZERO_GRADIENT)
            int_index = clip(math.to_int32(float_idx), 0, wrap(spatial(self._surf_index), '(x,y,z)')-1)
            face_index = self._surf_index[int_index]
        else:
            surf_float_idx = (surface_pos - self._bounds.lower) / self.size * self.resolution
            normal = math.grid_sample(self._grad, surf_float_idx - 1, math.extrapolation.ZERO_GRADIENT)
            # normal = math.where(self._bounds.lies_inside(surface_pos), normal, sdf_grad)  # use current normal if surface point is outside SDF grid
            normal = math.vec_normalize(normal)
            face_index = None
        offset = normal.vector @ surface_pos.vector
        return sgn_dist, to_surf, normal, offset, face_index

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        float_idx = (location - self._bounds.lower) / self.size * self.resolution
        sdf_val = math.grid_sample(self._sdf, float_idx - .5, math.extrapolation.ZERO_GRADIENT)
        if self._approximate_outside:
            within_bounds = self._bounds.lies_inside(location)
            dist_from_center = math.vec_length(location - self._center) - self._bounding_radius
            return math.where(within_bounds, sdf_val, dist_from_center)
        else:
            return sdf_val

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self._bounding_radius

    def bounding_half_extent(self) -> Tensor:
        return self._bounds.half_size  # this could be too small if the center is not in the middle of the bounds

    def shifted(self, delta: Tensor) -> 'Geometry':
        return SDFGrid(self._sdf, self._bounds.shifted(delta), self._approximate_outside, self._grad, self._to_surface, self._surf_normal, self._surf_index, self._center + delta, self._volume, self._bounding_radius)

    def at(self, center: Tensor) -> 'Geometry':
        return self.shifted(center - self._center)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support rotation")

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        off_center = self._center - self._bounds.center
        volume = self._volume * factor ** self.spatial_rank
        bounds = self._bounds.scaled(factor).shifted(off_center * (factor - 1)).corner_representation()
        return SDFGrid(self._sdf, bounds, self._approximate_outside, self._grad, self._to_surface, self._surf_normal, self._surf_index, self._center, volume, self._bounding_radius * factor)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if 'vector' in item:
            raise NotImplementedError("SDF projection not yet supported")
        return SDFGrid(self._sdf[item], self._bounds[item], self._approximate_outside, math.slice(self._grad, item), math.slice(self._to_surface, item), math.slice(self._surf_normal, item), math.slice(self._surf_index, item), math.slice(self._center, item), math.slice(self._volume, item), math.slice(self._bounding_radius, item))

    def approximate_occupancy(self):
        assert self._surf_normal is not None
        unit_corners = Cuboid(half_size=.5*self.dx).corners
        surf_dist = self._surf_normal.vector @ self._to_surface.vector
        corner_sdf = unit_corners.vector @ self._surf_normal.vector - surf_dist
        total_dist = math.sum(abs(corner_sdf), dual)
        neg_dist = math.sum(math.minimum(0, corner_sdf), dual)
        occ_near_surf = -neg_dist / total_dist
        occ_away = self._sdf <= 0
        return math.where(abs(self._sdf) < .5*math.vec_length(self.dx), occ_near_surf, occ_away)

    def approximate_fraction_inside(self, other_geometry: 'Geometry', balance: Union[Tensor, Number] = 0.5) -> Tensor:
        if other_geometry == self.grid and math.always_close(balance, .5):
            return self.approximate_occupancy()
        else:
            return Geometry.approximate_fraction_inside(self, other_geometry, balance)

    def downsample2x(self):
        s, g, t, n, i = [math.downsample2x(v) for v in (self._sdf, self._grad, self._to_surface, self._surf_normal, self._surf_index)]
        return SDFGrid(s, self._bounds, self._approximate_outside, g, t, n, i, self._center, self._volume, self._bounding_radius)


def sample_sdf(geometry: Geometry,
               bounds: BaseBox | UniformGrid = None,
               resolution: Shape = math.EMPTY_SHAPE,
               approximate_outside=False,
               rebuild: Optional[str] = None,
               valid_dist=None,
               rel_margin=.1,
               abs_margin=0.,
               cache_surface=False,
               **resolution_: int) -> SDFGrid:
    """
    Build a grid of signed distance values for a given `Geometry` object.

    Args:
        geometry: `Geometry` to capture.
        bounds: Grid limits in world space.
        resolution: Grid resolution.
        **resolution_: Grid resolution as `kwargs`, e.g. `x=64, y=32`.
        approximate_outside: Whether queries outside the SDF grid should return approximate values. This requires additional computations.
        rebuild: If `'from-surface'`, SDF values are calculated from a narrow strip above the enclosed surface. This is more accurate but requires additional steps.
            If `None` (default), SDF values are queried from `geometry`.
            `'auto'` rebuilds when geometry querying is expected to be in accurate.

    Returns:
        SDF grid as `Geometry`.
    """
    resolution = resolution & spatial(**resolution_)
    if bounds is None:
        bounds: BaseBox = geometry.bounding_box()
        bounds = Cuboid(bounds.center, half_size=bounds.half_size * (1 + 2 * rel_margin) + 2 * abs_margin)
    elif isinstance(bounds, UniformGrid):
        assert not resolution, f"When specifying a UniformGrid, separate resolution values are not allowed."
        resolution = bounds.resolution
        bounds = bounds.bounds
    points = UniformGrid(resolution, bounds).center
    reduce = instance(geometry) & spatial(geometry)
    if reduce:
        center = math.mean(geometry.center, reduce)
        volume = None
        bounding_radius = None
        rebuild = 'from-surface' if rebuild == 'auto' else rebuild
    else:
        center = geometry.center
        volume = geometry.volume
        bounding_radius = geometry.bounding_radius()
        rebuild = None if rebuild == 'auto' else rebuild
    if cache_surface or rebuild is not None:
        sdf, delta, normal, _, idx = geometry.approximate_closest_surface(points)
        approximate = SDFGrid(sdf, bounds, approximate_outside, None, delta, normal, idx, center=center, volume=volume, bounding_radius=bounding_radius)
    else:
        sdf = geometry.approximate_signed_distance(points)
        approximate = SDFGrid(sdf, bounds, approximate_outside, center=center, volume=volume, bounding_radius=bounding_radius)
    if rebuild is None:
        return approximate
    assert rebuild in ['from-surface']
    dx = bounds.size / resolution
    min_dist = math.sum(dx ** 2) ** (1 / geometry.spatial_rank)
    valid_dist = math.maximum(min_dist, valid_dist) if valid_dist is not None else min_dist
    sdf = rebuild_sdf(approximate, 0, valid_dist, refine=[geometry])
    return SDFGrid(sdf, bounds, approximate_outside, center=center, volume=volume, bounding_radius=bounding_radius)


def rebuild_sdf(sdf: SDFGrid, min_level=None, max_level=None, step_count: int = None, refine: Sequence[Geometry] = ()) -> Tensor:
    sample_points = sdf.points
    dist0, delta, *_ = sdf.approximate_closest_surface(sample_points)
    closest = sample_points + delta
    closest = math.where((sdf.values >= min_level) & (sdf.values <= max_level), closest, math.NAN)
    for _ in range(step_count if step_count is not None else sum(sdf.resolution.sizes)):
        abs_dist = math.vec_length(closest - sample_points)
        abs_dist = math.where(math.is_finite(abs_dist), abs_dist, math.INF)
        if step_count is None and math.all(math.is_finite(abs_dist)):
            break
        closest_nb = math.at_min_neighbor(closest, key_grid=abs_dist, padding=math.INF, offsets=(-1, 0, 1), diagonal=False)
        closest = math.where(math.is_finite(abs_dist), closest, closest_nb)
    for geo in refine:
        closest = refine_closest(sample_points, closest, geo, max_level)
    dist = math.vec_length(closest - sample_points) * math.sign(dist0)
    return dist


def refine_closest(sample_points, closest, refine: Geometry, max_step, steps=10):
    # trj = [closest]
    for ref_step in range(steps):
        sgn_dist, delta, normal, offset, _ = refine.approximate_closest_surface(closest)
        tang_proj = sample_points - normal * (normal.vector @ sample_points.vector - offset)
        walk_on_surface = clip_length(tang_proj - closest, 0, max_step * min(1, .5 ** (ref_step - steps / 2)))
        better_closest = (closest + delta) + walk_on_surface
        closest = math.where(refine.lies_inside(better_closest), closest, better_closest)  # don't walk into negative SDF
        # trj.append(closest)
    # from phi.vis import plot
    # from phi.field import PointCloud
    # plot(PointCloud(sample_points, stack(trj, batch('t')) - sample_points), animate='t', frame_time=250)
    _, delta, *_ = refine.approximate_closest_surface(closest)
    closest += delta
    return closest