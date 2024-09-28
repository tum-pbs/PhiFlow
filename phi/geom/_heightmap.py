from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any

import numpy

from phiml import math
from phiml.math import Tensor, spatial, unstack, stack, vec, wrap, Shape, channel, concat, instance, batch, rename_dims, shape, non_channel, expand
from phiml.math._nd import index_shift_widths
from phiml.math._tensors import cached
from phiml.math.magic import slicing_dict
from . import UniformGrid
from ._box import Box
from ._functions import normal_from_slope, y_intersect_2d, plane_sgn_dist
from ._geom import Geometry


class Heightmap(Geometry):

    def __init__(self,
                 height: Tensor,
                 bounds: Box,
                 max_dist: Union[float, Tensor],
                 fill_below: Union[bool, Tensor] = True,
                 extrapolation: Union[float, str, math.Extrapolation] = None,
                 faces=None):
        """

        Args:
            height: Heightmap `Tensor` of absolute (world-space) height values.
                Scalar height values on a d-1 dimensional grid.
            bounds: d-dimensional bounds.
                Locations outside `bounds' can never lie inside this geometry if `extrapolation is None`.
                Otherwise, only the height dimension is checked.
                The grid dimensions of `bounds` must be finite but the height dimension may be infinite to count all values above/below `height` as inside.
            max_dist: Maximum distance up to which the distance approximations should be valid.
                This does not affect the number of computations performed to compute the distance.
                Low values increase accuracy close to the surface but trade off possibly very wrong distances further away.
            fill_below: Whether the inside is below or above the height values.
            extrapolation: Surface height outside `bounds´. Can be any valid `phiml.math.Extrapolation`, such as a constant.
                If not `None`, values outside `bounds` will be checked against the extrapolated `height` values.
                Otherwise, values outside `bounds` always lie on the outside.
        """
        assert channel(height).is_empty, f"height must be a scalar quantity but got {height.shape}"
        assert spatial(height), f"height field must have at least one spatial dim but got {height}"
        assert bounds.vector.size == spatial(height).rank + 1, f"bounds must include the spatial grid dimensions {spatial(height)} and the height dimension but got {bounds}"
        dims = bounds.vector.item_names
        self._hdim = spatial(*dims).without(height.shape).name
        if math.all_available(height, bounds.lower, bounds.upper):
            assert bounds[self._hdim].lies_inside(height).all, f"All height values should be within the {self._hdim}-range given by bounds but got height={height}"
        self._height = height
        self._fill_below = wrap(fill_below)
        self._bounds = bounds
        self._max_dist = wrap(max_dist)
        self._extrapolation = math.as_extrapolation(extrapolation)
        if faces is None:
            proj_faces = build_faces(self)
            with numpy.errstate(divide='ignore', invalid='ignore'):
                secondary_idx = math.map(find_most_important_neighbor, proj_faces, self.dx, self.resolution, self._hdim, self._fill_below, self._max_dist, dims=instance, unwrap_scalars=False)
                secondary_faces = math.map(math.gather, proj_faces, secondary_idx, dims=instance)
            self._faces: Face = stack([proj_faces, *unstack(secondary_faces, 'side')], batch(consider='self,outside,inside'), expand_values=True)
            self._faces = cached(self._faces)  # otherwise, this may get expanded during tracing
        else:
            self._faces = faces

    @property
    def height(self):
        return self._height

    @property
    def bounds(self):
        return self._bounds

    @property
    def max_dist(self):
        return self._max_dist

    @property
    def fill_below(self):
        return self._fill_below

    @property
    def extrapolation(self):
        return self._extrapolation

    @property
    def shape(self) -> Shape:
        return (self._height.shape - 1) & channel(self._bounds)

    @property
    def resolution(self):
        return spatial(self._height) - 1

    @property
    def grid_bounds(self):
        return self._bounds[self.resolution.name_list]

    @property
    def up(self):
        dims = self._bounds.vector.item_names
        height_unit = vec(**{d: 1 if d == self._hdim else 0 for d in dims})
        return math.where(self._fill_below, height_unit, -height_unit)

    @property
    def dx(self):
        return self._bounds.size[self.resolution.name_list] / spatial(self.resolution)

    @property
    def vertices(self):
        hdim = self._hdim
        space = self.vector.item_names
        pos = self.grid_bounds.local_to_global(math.meshgrid(spatial(self._height)) / self.resolution)
        vert = stack({dim: self.height if dim == hdim else pos[dim] for dim in space}, channel('vector'))
        return vert

    def lies_inside(self, location: Tensor) -> Tensor:
        location = rename_dims(location, self.resolution.names, ['loc_' + n for n in self.resolution.names])
        projected_loc = location[self.resolution.name_list]
        @math.map_i2b
        def lies_inside_(height, grid_bounds, bounds, fill_below, extrapolation):
            float_idx = (projected_loc - grid_bounds.lower) / grid_bounds.size * self.resolution
            if extrapolation is None:
                within_bounds = bounds.lies_inside(location)
            else:
                within_bounds = bounds[self._hdim].lies_inside(location[self._hdim])
            surface_height = math.grid_sample(height, float_idx - 1, math.NAN if extrapolation is None else extrapolation)
            is_below = location[self._hdim] <= surface_height
            inside = is_below == fill_below
            result = math.where(within_bounds, inside, False)
            return rename_dims(result, ['loc_' + n for n in self.resolution.names], self.resolution.names)
        return math.any(lies_inside_(self._height, self.grid_bounds, self._bounds, self._fill_below, self._extrapolation), instance(self))

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        grid_bounds = math.i2b(self.grid_bounds)
        faces = math.i2b(self._faces)
        cell_idx = cell_index(location, grid_bounds, self.resolution, clip=True)
        # --- gather face infos at projected cell ---
        normals = faces.normal[cell_idx]
        offsets = faces.origin_distance[cell_idx]
        face_idx = faces.index[cell_idx]
        # --- test location against all considered faces and boundaries ---
        # distances = plane_sgn_dist(-offsets, normals, location)  # offset has the - convention here
        distances = normals.vector @ location.vector + offsets
        projected_onto_face = location - normals * distances
        projected_idx = cell_index(projected_onto_face, grid_bounds, self.resolution, clip=False)
        projects_onto_face = math.all(projected_idx == face_idx, channel)
        proj_delta = normals * -distances
        # --- if not projected onto face, use distance to highest point instead ---
        delta_highest = faces.extrema_points[cell_idx] - location
        flat_normal = math.vec_normalize(normals[self.resolution.name_list], epsilon=1e-5)
        delta_edge = flat_normal * (delta_highest[self.resolution].vector @ flat_normal.vector)  # project onto flat normal
        delta_edge = concat([delta_edge, delta_highest[[self._hdim]]], 'vector')
        distance_edge = math.vec_length(delta_edge, eps=1e-5)
        delta_highest, distance_edge = math.at_min((delta_highest, distance_edge), distance_edge, 'extremum')
        distance_edge = math.where(distances < 0, -distance_edge, distance_edge)  # copy sign of distances onto distance_edges to always return the signed distance
        distances = math.where(projects_onto_face, distances, distance_edge)
        # --- use closest face from considered ---
        delta = math.where(projects_onto_face, proj_delta, delta_highest)
        return math.at_min((distances, delta, normals, offsets, face_idx), key=abs(distances), dim=batch('consider') & instance(self).as_batch())

    def shallow_equals(self, other):
        return self == other

    def __repr__(self):
        return f"Heightmap {self.resolution}, bounds={self._bounds}"

    def __variable_attrs__(self):
        return '_height', '_bounds', '_max_dist', '_fill_below', '_extrapolation', '_faces'

    def __value_attrs__(self):
        return ()

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return Heightmap(self._height[item], self._bounds[item], self._max_dist[item], self._fill_below[item], self._extrapolation[item] if self._extrapolation is not None else None, math.slice(self._faces, item))

    def bounding_half_extent(self) -> Tensor:
        h_min, h_max = self._faces.extrema_points[{'consider': 0, 'vector': self._hdim}].extremum
        dh = h_max - h_min
        return stack({d: self.dx[d] if d in self.resolution else dh for d in self.vector.item_names}, channel('vector'), expand_values=True) * .5

    @property
    def center(self) -> Tensor:
        return self._faces.center.consider[0]

    @property
    def volume(self) -> Tensor:
        return math.prod(self.bounding_half_extent() * 2, channel)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_centers(self) -> Tensor:
        return self._faces.center

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_normals(self) -> Tensor:
        return self._faces.normal

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return non_channel(self._faces.center)

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return self.approximate_closest_surface(location)[0]

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self._bounds.bounding_radius()

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError


@dataclass
class Face:
    center: Tensor
    normal: Tensor
    extrema_points: Tensor
    origin_distance: Tensor
    index: Tensor

    def distance_to(self, location: Tensor):
        return self.normal.vector @ location.vector + self.origin_distance


def build_faces(heightmap: Heightmap):
    height = heightmap.height
    flat_space = spatial(height).name_list
    pos = heightmap.vertices
    center = math.neighbor_mean(pos)
    face_slopes = {}
    for dim in flat_space:
        edge_slope = math.spatial_gradient(height, 1, 'forward', None, dim, None)
        face_slope_dim = math.neighbor_mean(edge_slope, [d for d in flat_space if d != dim])
        face_slopes[dim] = face_slope_dim
    face_slope = vec(**face_slopes)
    face_slope /= heightmap.dx
    highest_point = math.at_max_neighbor(pos, height, spatial)
    lowest_point = math.at_min_neighbor(pos, height, spatial)
    extrema_points = stack({'lowest': lowest_point, 'highest': highest_point}, batch('extremum'))
    face_n, face_d = plane_from_slope(face_slope, center)
    negate_below = math.where(heightmap._fill_below, 1, -1)
    face_n *= negate_below
    face_d *= negate_below
    index = math.meshgrid(heightmap.resolution)
    return Face(center, face_n, extrema_points, face_d, index)


def plane_from_slope(slope: Tensor, point: Tensor):
    space = channel(point).item_names[0]
    normal = normal_from_slope(slope, space)
    origin_distance = (-point).vector @ normal.vector  # ToDo we should really use the positive sign
    return normal, origin_distance


def find_most_important_neighbor(face: Face, dx, resolution, hdim, fill_below, max_dist):
    outside = vec(batch('side'), outside=True, inside=False)
    above = wrap(outside) ^ ~fill_below
    normals = face.normal.type['face']
    center = face.center
    extrema_points = face.extrema_points
    flat_space = resolution.name_list
    # --- find close-by face that could be closer to points over self face ---
    max_cells = math.maximum(1, math.to_int32(max_dist / dx))
    offsets = []
    errors = []
    with math.NUMPY:
        shifts = math.meshgrid(spatial(**(max_cells * 2 + 1).vector)) - max_cells
    for i in unstack(shifts, spatial):
        flat_delta = i * dx
        flat_dist = math.vec_length(flat_delta)
        if (i != 0).any and ((flat_dist <= max_dist).all or (math.vec_squared(i) <= 1.5).all) and (abs(i) < resolution).all:
            self_normals, other_normals = math.index_shift(normals, (0, i), padding=None)
            self_center, other_center = math.index_shift(center, (0, i), padding=None)
            _, other_extrema = math.index_shift(extrema_points, (0, i), padding=None)
            # dist_to_lowest = math.vec_length(self_center[flat_space] - other_lowest[flat_space])
            # dist_to_highest = math.vec_length(self_center[flat_space] - other_highest[flat_space])
            dist_to_extrema = math.vec_length(self_center[flat_space] - other_extrema[flat_space])
            flat_component = flat_delta.vector @ other_normals[flat_space].vector
            up_component = other_normals[hdim]
            heights_over_self, dists = y_intersect_2d(up_component, flat_component, dist_to_extrema, other_extrema[hdim] - self_center[hdim])
            start_height, start_dist = math.at_min((heights_over_self, dists), key=abs(heights_over_self), dim='extremum')
            mid_height, mid_dist = math.at_max((heights_over_self, dists), key=abs(heights_over_self), dim='extremum')
            has_intersection = (mid_height > 0) == above
            # --- integrate column above face: linearly increasing + constant ---
            start_height, mid_height = abs(start_height), abs(mid_height)
            mid_error = math.maximum(0, mid_height - mid_dist)
            start_height_bounded = math.minimum(start_height, max_dist)
            mid_height_bounded = math.minimum(mid_height, max_dist)  # the error increases linearly up to this height, then assume constant error
            error1 = .25 * start_height_bounded * mid_error * (start_height_bounded / mid_height)  # 1/4 error up to first intersection
            error2 = .5 * (mid_height_bounded - start_height_bounded) * mid_error * (mid_height_bounded / mid_height)  # 1/2 error up to second intersection
            error3 = mid_error * (max_dist - mid_height_bounded)  #  constant error until max_dist
            error = error1 + error2 + error3
            error = math.where(has_intersection, error, 0)
            offsets.append(i)
            error_with_edge = math.pad(error, index_shift_widths((0, i))[0], -1)
            errors.append(error_with_edge)
            # print(f"for offset {i} errors: {error_with_edge:row}")
    best_offset = math.at_max(stack(offsets, instance('neighbors')), stack(errors, instance('neighbors')), 'neighbors')
    best_neighbor = best_offset + face.index
    return best_neighbor


def cell_index(location, grid_bounds, resolution, clip=True):
    projected_loc = location[resolution]
    flat_rel = (projected_loc - grid_bounds.lower) / grid_bounds.size
    cell_idx = math.to_int32(math.floor(flat_rel * resolution))
    if clip:
        cell_idx = math.clip(cell_idx, 0, vec(**(resolution - 1).untyped_dict))
    return cell_idx
