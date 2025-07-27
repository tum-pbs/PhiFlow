from dataclasses import dataclass
from functools import cached_property
from typing import Union, Tuple, Dict

from phiml import math
from phiml.dataclasses import replace, sliceable, data_eq
from phiml.math import Tensor, Shape, spatial, dual, dprod, stack, dmax, dmin, dmean, wrap, linspace, PI, where, concat, clip, dpack, \
    mean, arcsin, maximum, instance, batch, map_types, unstack, expand, to_int32, vec, meshgrid, minimum, dsum, sign, channel
from phiml.math.extrapolation import ZERO_GRADIENT
from . import Mesh, Sphere, Cylinder
from ._functions import vec_length, cross, vec_normalize, rotate_vector, sample_helix, orthogonal_vector, closest_normal_vector, clip_length
from ._geom import Geometry
from ._mesh_builder import MeshBuilder
from ._spline import closest_param, spline_eval


@data_eq
@sliceable(keepdims='vector')
@dataclass(frozen=True)
class SplineSolid(Geometry):
    """
    Internal coordinates (u,v) are in the range [0, N] where N is the number of points along that axis.
    """

    points: Tensor
    """2D spatial tensor of points"""
    thickness: Tensor
    """2D spatial tensor matching vertices"""
    fillet: Dict[str, Tensor]
    """Roundness by vertex by boundary, such as 'u-', 'u+', 'v-', 'v+'. 1 puts a full cylinder at the edge, 0 is a sharp edge."""
    order: Dict[str, int]
    """Spline order along each axis, e.g. {'u': 1, 'v': 2}"""

    variable_attrs: Tuple[str, ...] = ('points', 'thickness', 'fillet')
    value_attrs: Tuple[str, ...] = ()

    def __post_init__(self):
        assert 'vector' in self.points.shape
        for dim, o in self.order.items():
            assert f'{dim}-' in self.fillet and f'{dim}+' in self.fillet
            assert dim in spatial(self.points)
            assert o < self.points.shape.get_size(dim), f"Spline order must be at must points-1 per dimension but got {o} for {dim} which has {self.points.shape.get_size(dim)} points."

    @property
    def shape(self) -> Shape:
        return self.points.shape & self.thickness.shape & batch(self.fillet)

    @property
    def resolution(self):
        return self.points.shape.spatial

    @cached_property
    def center(self) -> Tensor:
        lin_center = math.neighbor_mean(self.points, spatial, None)
        return lin_center

    @cached_property
    def radius(self):
        return .5 * self.thickness

    @cached_property
    def volume(self) -> Tensor:
        dx = math.spatial_gradient(self.points, difference='forward', stack_dim=dual('gradient'))[{d: slice(0, -1) for d in spatial(self.points)}]
        return dprod(vec_length(dx))

    @cached_property
    def corner_shape(self) -> Shape:
        return spatial(self.points).as_dual().with_sizes('lo,up') + (spatial(self.points) - 1)

    @cached_property
    def corners(self) -> Tensor:
        result = [self.points[{dim[1:]: slice(o, None if o else -1) for dim, o in offset.items()}] for offset in self.corner_shape.dual.meshgrid()]
        return stack(result, self.corner_shape.dual)

    @cached_property
    def corner_radii(self) -> Tensor:
        result = [self.thickness[{dim[1:]: slice(o, None if o else -1) for dim, o in offset.items()}] for offset in self.corner_shape.dual.meshgrid()]
        return stack(result, self.corner_shape.dual)

    @cached_property
    def _central_tangents(self):
        tangents = {dim.name[1:]: dmean(self.corners[{dim: 1}] - self.corners[{dim: 0}]) for dim in self.corner_shape.dual}
        return stack(tangents, '~tangents')

    @cached_property
    def _central_uv_extent(self):
        tangents = {dim.name[1:]: vec_length(dmean(self.corners[{dim: 1}] - self.corners[{dim: 0}])) for dim in self.corner_shape.dual}
        return stack(tangents, '~tangents')

    @cached_property
    def _central_point_tangents(self):
        return math.neighbor_mean(self._central_tangents, spatial, ZERO_GRADIENT, extend_bounds=(1, 0))

    @cached_property
    def _central_point_normals(self):
        t1, t2 = self._central_point_tangents.tangents.dual
        return vec_normalize(cross(t1, t2))

    @cached_property
    def _surface_points(self):
        front_back = wrap([-1, 1], dual(side='front,back'))
        return self.points + front_back * self.radius * self._central_point_normals

    @cached_property
    def _surface_corners(self):
        result = [self._surface_points[{dim[1:]: slice(o, None if o else -1) for dim, o in offset.items()}] for offset in self.corner_shape.dual.meshgrid()]
        return stack(result, self.corner_shape.dual)

    @cached_property
    def _surface_point_tangents(self):
        tangents = {dim.name[1:]: mean(self._surface_corners[{dim: 1}] - self._surface_corners[{dim: 0}], self.corner_shape.dual) for dim in self.corner_shape.dual}
        tangents = stack(tangents, '~tangents')
        return math.neighbor_mean(tangents, spatial, ZERO_GRADIENT, extend_bounds=(1, 0))

    @cached_property
    def _surface_point_normals(self):
        front_back = wrap([1, -1], dual(side='front,back'))
        t1, t2 = self._surface_point_tangents.tangents.dual
        return vec_normalize(cross(t1, t2)) * -front_back

    def surface_mesh(self, vertex_spacing: Union[float, Tensor] = None, min_cyl_segments=5, min_corner_segments=2, merge_instances=False, displacement=None):  # mesh can get self-intersections for larger corner_segments
        mb = MeshBuilder(2, batch(self) if merge_instances else batch(self) + instance(self), source_face_shape=self.face_shape if displacement is not None else None)
        self.build_surface_mesh(mb, vertex_spacing, min_cyl_segments, min_corner_segments)
        if displacement is not None:
            return mb.build_displaced_mesh(displacement)
        return mb.build_mesh()

    def sample_surface(self, vertex_spacing: Union[float, Tensor] = None, dim: Shape = instance('surf_points'), min_cyl_segments=5, min_corner_segments=2, merge_instances=False) -> Tensor:
        mb = MeshBuilder(2, batch(self) if merge_instances else batch(self) + instance(self), source_face_shape=self.face_shape)
        self.build_surface_mesh(mb, vertex_spacing, min_cyl_segments, min_corner_segments)
        return mb.sample_surface(dim)

    def build_surface_mesh(self, mb: MeshBuilder, vertex_spacing: Union[float, Tensor], min_cyl_segments=5, min_corner_segments=2):  # mesh can get self-intersections for larger corner_segments
        front_back = wrap([1, -1], dual(side='front,back'))
        v, u = spatial(self.points).names
        vs, us = spatial(self.points).sizes
        # --- Spline segments ---
        if vertex_spacing is None:
            points = self.points
            tangents = self._central_point_tangents
            normals = self._central_point_normals
            surface_normals = self._surface_point_normals
            fillet_dict = self.fillet
            radius = self.radius
            resolution = self.resolution
            idx = meshgrid(front_back.shape + (self.resolution-1), 'index') + vec('index', **{'~side': 0, v: 1, u: 1})
            segment_idx = mb.new_quads('spline', points + radius * surface_normals, idx, flip=front_back > 0)
        else:
            v1, v2, v3, v4 = unstack(self.corners, dual)
            len_v = .5 * (vec_length(v2 - v1) + vec_length(v4 - v3))
            len_u = .5 * (vec_length(v3 - v1) + vec_length(v4 - v2))
            res_v = maximum(1, math.to_int32(math.round(len_v / vertex_spacing)))
            res_u = maximum(1, math.to_int32(math.round(len_u / vertex_spacing)))
            res_v, res_u = res_v.max, res_u.max
            interp = meshgrid(**{v: int(res_v), u: int(res_u)}) / (res_v-1, res_u-1)
            points = math.grid_sample(self.points, interp, ZERO_GRADIENT)
            thickness = self.thickness_at(interp)
            subdivided = SplineSolid(points, thickness, self.fillet, self.order)
            tangents = subdivided._central_point_tangents
            normals = subdivided._central_point_normals
            surface_normals = subdivided._surface_point_normals
            resolution = spatial(points)
            fillet_dict = subdivided.fillet
            radius = .5 * thickness
            idx = vec('index', **{'~side': meshgrid(front_back.shape, None), v: 1, u: 1})
            segment_idx = mb.new_quads('spline', points + radius * surface_normals, idx, flip=front_back > 0)
        # --- Rounded edges ---
        for edge_, fillet in fillet_dict.items():
            edge, is_upper = edge_[:-1], edge_[-1] == '+'
            other_edge = spatial(points).without(edge).name
            idx = vec(**{edge: resolution.get_size(edge) if is_upper else 0, other_edge: meshgrid(resolution[other_edge]-1)+1, '~side': meshgrid(front_back.shape)})
            fillet = clip(fillet, 1e-5, .99)
            edge_points = points[{edge: -1 if is_upper else 0}]
            edge_tangent_c = vec_normalize(tangents[{'~tangents': other_edge, edge: -1 if is_upper else 0}])
            edge_tangent = edge_tangent_c  # ToDo rotate to match thickness change along `edge`
            edge_normals = surface_normals[{edge: -1 if is_upper else 0}]
            edge_normals_c = normals[{edge: -1 if is_upper else 0}]
            edge_rad = radius[{edge: -1 if is_upper else 0}]
            cyl_center = edge_points + edge_normals * edge_rad * (1-fillet)
            incline_angle = arcsin(cross(edge_normals, edge_normals_c).vector @ edge_tangent_c.vector) * (-1 if is_upper else 1)
            angle = linspace(0, PI/2 - incline_angle, spatial(cyl=min_cyl_segments+1)).cyl[1:]
            cv = cyl_center + rotate_vector(fillet * edge_rad * edge_normals, edge_tangent * angle * front_back * where(is_upper ^ (edge == v), 1, -1))
            e_idx = mb.add_vertices(edge_, cv)
            ec_idx = mb.add_vertices(edge_+'-center', .5 * (cv.cyl[-1]['front'] + cv.cyl[-1]['back']))
            mb.add_quads(concat([segment_idx[{edge: -1 if is_upper else 0}], e_idx, ec_idx], 'cyl', expand_values=True), idx, flip=(front_back > 0) ^ is_upper ^ (edge != v))
        # --- Rounded corners ---
        for u_up in '-+':
            for v_up in '-+':
                idx = vec(**{u: us if u_up == '+' else 0, v: vs if v_up == '+' else 0, '~side': meshgrid(front_back.shape)})
                u_verts = mb.vertices(f"{u}{u_up}")[{v: 0 if v_up == '-' else -1}]
                v_verts = mb.vertices(f"{v}{v_up}")[{u: 0 if u_up == '-' else -1}]
                normal = surface_normals[{u: 0 if u_up == '-' else -1, v: 0 if v_up == '-' else -1}]
                corner = points[{u: 0 if u_up == '-' else -1, v: 0 if v_up == '-' else -1}]
                t = linspace(0, 1, spatial(corner=min_corner_segments+1)).corner[1:-1]
                corner_points_round = sample_helix(corner, normal, u_verts, v_verts, t)
                corner_points_straight = t * u_verts + (1-t) * v_verts
                fillet_u = fillet_dict[f"{u}{u_up}"][{v: 0 if v_up == '-' else -1}]
                fillet_v = fillet_dict[f"{v}{v_up}"][{u: 0 if u_up == '-' else -1}]
                corner_roundness = fillet_u * fillet_v
                corner_points = (1-corner_roundness) * corner_points_straight + corner_roundness * corner_points_round
                c_idx = mb.add_vertices(f"{u}{u_up},{v}{v_up}", corner_points)
                u_idx = mb.vertex_indices(f"{u}{u_up}")[{v: 0 if v_up == '-' else -1}]
                v_idx = mb.vertex_indices(f"{v}{v_up}")[{u: 0 if u_up == '-' else -1}]
                # --- Center edge ---
                cc_idx = mb.add_vertices(f"{u}{u_up},{v}{v_up}-center", .5 * (corner_points.cyl[-1]['front'] + corner_points.cyl[-1]['back']))
                uc_idx = mb.vertex_indices(f"{u}{u_up}-center")[{v: 0 if v_up == '-' else -1}]
                vc_idx = mb.vertex_indices(f"{v}{v_up}-center")[{u: 0 if u_up == '-' else -1}]
                cc_idx = concat([uc_idx, cc_idx, vc_idx], 'corner', expand_values=True)
                c_idx = concat([u_idx, c_idx, v_idx], 'corner', expand_values=True)
                c_idx = concat([c_idx, cc_idx], 'cyl', expand_values=True)
                mb.add_tris(segment_idx[{v: 0 if v_up == '-' else -1, u: 0 if u_up == '-' else -1}], c_idx.cyl[0], idx, flip=(front_back > 0) ^ (u_up == v_up))
                mb.add_quads(c_idx, idx, flip=(front_back > 0) ^ (u_up == v_up))
        # mb.debug_show(normals=True)

    def lies_inside(self, location: Tensor) -> Tensor:
        return self.approximate_signed_distance(location) <= 0

    def _center_mesh(self) -> Mesh:
        mb = MeshBuilder(2)
        mb.new_quads('spline', self.points, flip=True)
        return mb.build_mesh()

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sdfs, deltas, normals, offsets, face_indices, *_ = map_types(SplineSolid._closest_surface, instance(self), batch)(self, location)
        return math.min((sdfs, deltas, normals, offsets, face_indices), instance(self), key=sdfs)

    @staticmethod
    def _closest_surface(self, location: Tensor):
        on_skeleton, uv, unbounded_uv, tangents = closest_param(self.order, self.points, location, uv_gradient=False)
        delta = location - on_skeleton
        normal_c = vec_normalize(cross(*tangents.tangents.dual))
        thickness = self.thickness_at(uv)
        radius = .5 * thickness
        # --- Compute effective fillet (1 inside valid uv range) ---
        is_edge = stack({'-': unbounded_uv < uv, '+': unbounded_uv > uv}, '~side')
        fillets = stack({dim: stack({'-': self.fillet[dim+'-'], '+': self.fillet[dim+'+']}, '~side') for dim in spatial(self.points).names}, 'spline:c')
        fillet = where(is_edge, fillets, 1)
        h = normal_c.vector @ delta
        is_corner = math.sum(is_edge, 'spline,~side') >= 2
        # --- Corner: Constrain sphere center to plane orthogonal to large fillet ---
        large_fillet = math.max(fillets * is_edge, '~side,spline')
        small_fillet = math.min(fillet, '~side,spline')
        large_fillet_tangent = math.min(tangents, '~tangents', key=dsum(fillet * is_edge).spline.as_dual('tangents'))
        ortho_dir = vec_normalize(cross(normal_c, large_fillet_tangent))
        w = ortho_dir.vector @ (location - on_skeleton)
        safe_h = radius * (1 - large_fillet)
        # ToDo need safe_w after all
        h_over = maximum(0, abs(h) - safe_h) * sign(h)
        over = clip_length(vec('dir', w=w, h=h_over), max_len=radius * (large_fillet - small_fillet), vec_dim='dir')
        wh = vec('dir', w=0, h=clip(h, -safe_h, safe_h)) + over
        dirs = stack({'w': ortho_dir, 'h': normal_c}, '~dir')
        c_sphere_offset = dirs @ wh
        c_sphere_rad = radius * small_fillet
        # --- Edge/segment: Constrain sphere center to line orthogonal to spline skeleton ---
        sphere_rad = radius * dmin(fillet.spline.as_dual('tangents'))
        sphere_h_lim = radius - sphere_rad
        sphere_h = clip(h, -sphere_h_lim, sphere_h_lim)
        sphere_offset = normal_c * sphere_h
        # --- Construct sphere ---
        sphere_offset = where(is_corner, c_sphere_offset, sphere_offset)
        sphere_rad = where(is_corner, c_sphere_rad, sphere_rad)
        sphere = Sphere(center=on_skeleton + sphere_offset, radius=sphere_rad)
        face_index = vec('index', **to_int32(clip(unbounded_uv+1, 0, self.resolution)).spline, **{'~side': to_int32(h <= 0)})
        def sphere_sdf(sphere: Sphere, location: Tensor):
            return sphere.approximate_closest_surface(location)
        sdf, delta, normal, offset, _ = map_types(sphere_sdf, instance(location), batch)(sphere, location)
        return sdf, delta, normal, offset, face_index, on_skeleton, uv, unbounded_uv, tangents, normal_c, thickness, is_edge, is_corner

    def thickness_at(self, uv: Tensor):
        if not spatial(self.thickness):
            return self.thickness
        return math.grid_sample(self.thickness, uv, ZERO_GRADIENT)

    def center_at(self, uv: Tensor):
        return math.grid_sample(self.points, uv, ZERO_GRADIENT)

    @cached_property
    def _merged_surface_mesh(self):
        return self.surface_mesh(5, 3, merge_instances=True)

    @cached_property
    def _surface_mesh(self):
        return self.surface_mesh(None, 5, 3, merge_instances=False)

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return self.approximate_closest_surface(location)[0]

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return dmax(vec_length(self.corners - self.center) + self.corner_radii)

    def bounding_half_extent(self) -> Tensor:
        return .5 * (dmax(self.corners) - dmin(self.corners)) + dmax(self.corner_radii)

    def at(self, center: Tensor) -> 'SplineSolid':
        assert self.resolution in center.shape, f"NURBSSolid.at() must be given new vertex positions"
        return replace(self, points=center)

    def shifted(self, delta: Tensor) -> 'SplineSolid':
        return replace(self, points=self.points + delta)

    def rotated(self, angle: Union[float, Tensor]) -> 'SplineSolid':
        return replace(self, points=rotate_vector(self.points, angle))

    def scaled(self, factor: Union[float, Tensor]) -> 'SplineSolid':
        return replace(self, points=self.points * factor, thickness=self.thickness * factor)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_shape(self) -> Shape:
        return dual(side='front,back') + (self.resolution+1)

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError

    @cached_property
    def face_areas(self) -> Tensor:
        first_edge, second_edge = spatial(self.points).names
        # --- Inner areas as triangles ---
        v1, v2, v3, v4 = unstack(self.corners, dual)
        tri1 = .5 * vec_length(cross(v2-v1, v3-v1))
        tri2 = .5 * vec_length(cross(v4-v1, v3-v1))
        inner_areas = tri1 + tri2
        areas = [[], [inner_areas], []]
        # --- Edges ---
        for edge_, fillet in self.fillet.items():
            edge, is_upper = edge_[:-1], edge_[-1] == '+'
            other_edge = spatial(self.points).without(edge).name
            edge_points = self.points[{edge: -1 if is_upper else 0}]
            lengths = vec_length(edge_points[{other_edge: slice(1, None)}] - edge_points[{other_edge: slice(None, -1)}])
            edge_thickness = self.thickness[{edge: -1 if is_upper else 0}]
            mean_rad = .25 * (edge_thickness[{other_edge: slice(1, None)}] + edge_thickness[{other_edge: slice(None, -1)}])
            flat_area = (1-fillet) * mean_rad * lengths
            curved_area = fillet * mean_rad * (PI/2) * lengths
            area = expand(flat_area + curved_area, spatial(**{edge: 1}))
            if edge == first_edge:
                areas[2 if is_upper else 0].append(area)
            else:
                areas[1].insert(2 if is_upper else 0, area)
        # --- Corners ---
        corners = [
            (0, 0, {first_edge: 0, second_edge: 0}, first_edge+'-', second_edge+'-'),
            (0, 2, {first_edge: 0, second_edge: -1}, first_edge+'-', second_edge+'+'),
            (2, 0, {first_edge: -1, second_edge: 0}, first_edge+'+', second_edge+'-'),
            (2, 2, {first_edge: -1, second_edge: -1}, first_edge+'+', second_edge+'+'),
        ]
        for i, j, idx, f1, f2 in corners:
            rad = self.radius[idx]
            min_fillet = math.minimum(self.fillet[f1], self.fillet[f2])
            max_fillet = math.maximum(self.fillet[f1], self.fillet[f2])
            curved = (min_fillet*rad)**2 * (PI/2) + (1-min_fillet)*rad * (PI/4) * min_fillet*rad  # 1/8-sphere + half of 1/4-cylinder
            large_flat = (max_fillet * rad)**2 * (PI / 4) + (1-max_fillet)*max_fillet * rad**2  # 1/4-circle + rect
            small_flat = (min_fillet * rad)**2 * (PI / 4) + (1-min_fillet)*min_fillet * rad**2
            corner = expand(curved + large_flat - small_flat, spatial(self.points).with_sizes(1))
            areas[i].insert(j, corner)
        result = concat([concat(a, second_edge) for a in areas], first_edge)
        return expand(result, dual(side='front,back'))

    @cached_property
    def face_fillet(self):
        """Mean fillet by face. 1 for spline segment faces."""
        un, vn = spatial(self.points).names
        u, v = spatial(self.points)-1
        u1, v1 = spatial(self.points).with_sizes(1)
        result = [[], [math.ones(self.resolution-1)], []]
        # --- Edges ---
        for edge_, fillet in self.fillet.items():
            edge, is_upper = edge_[:-1], edge_[-1] == '+'
            if edge == un:
                result[2 if is_upper else 0].append(expand(fillet, v, u1))
            else:
                result[1].insert(2 if is_upper else 0, expand(fillet, v1, u))
        # --- Corners ---
        corners = [
            (0, 0, {un: 0, vn: 0}, un+'-', vn+'-'),
            (0, 2, {un: 0, vn: -1}, un+'-', vn+'+'),
            (2, 0, {un: -1, vn: 0}, un+'+', vn+'-'),
            (2, 2, {un: -1, vn: -1}, un+'+', vn+'+'),
        ]
        for i, j, idx, f1, f2 in corners:
            fillet1, fillet2 = self.fillet[f1], self.fillet[f2]
            result[i].insert(j, expand(minimum(fillet1, fillet2), v1, u1))
        result = concat([concat(r, vn, expand_values=True) for r in result], un, expand_values=True)
        return expand(result, dual(side='front,back'))


    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[str, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        v, u = spatial(self.points).names
        return {
            u+'-': {u: slice(0, 1)},
            u+'+': {u: slice(-1, None)},
            v+'-': {v: slice(0, 1)},
            v+'+': {v: slice(-1, None)},
        }

    def __mul__(self, other):
        if isinstance(other, (float, Tensor)):
            return replace(self, points=self.points * other, thickness=self.thickness * other, fillet={k: v * other for k, v in self.fillet.items()})
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, SplineSolid):
            return replace(self, points=self.points + other.points, thickness=self.thickness + other.thickness, fillet={k: v + other.fillet[k] for k, v in self.fillet.items()})
        return NotImplemented


def to_spline(geo: Geometry, /, per_vertex_thickness=True, variable_attrs=('points', 'thickness', 'fillet'), value_attrs=(), rel_separation=1e-5) -> SplineSolid:
    assert geo.spatial_rank == 3, f"SplineSolid must be fit to 3D geometry but got {geo}"
    zero = expand(wrap(0.), instance(geo))
    one = expand(wrap(1.), instance(geo))
    if isinstance(geo, Cylinder):
        tips = dpack(geo.face_centers['bottom,top'], 'u:s')
        right = orthogonal_vector(geo.up)
        side_eps = geo.depth * rel_separation * right * wrap([-1, 1], 'v:s')
        points = tips + side_eps
        thickness = geo.radius * 2
        if per_vertex_thickness:
            thickness = expand(thickness, spatial(points))
        return SplineSolid(points, thickness, {'u-': zero, 'u+': zero, 'v-': one, 'v+': one}, {'u': 1, 'v': 1}, variable_attrs, value_attrs)
    elif isinstance(geo, Box):
        thickness, th_idx = math.min((geo.size, range), 'vector', key=geo.size)
        u_idx, v_idx = (th_idx + 1) % 3, (th_idx + 2) % 3
        vu, vv = geo.rotation_matrix[{'~vector': u_idx}], geo.rotation_matrix[{'~vector': v_idx}]
        su, sv = geo.size.vector[u_idx], geo.size.vector[v_idx]
        u = vu * su * wrap([-.5, .5], 'u:s')
        v = vv * sv * wrap([-.5, .5], 'v:s')
        points = geo.center + u + v
        if per_vertex_thickness:
            thickness = expand(thickness, spatial(points))
        return SplineSolid(points, thickness, {'u-': zero, 'u+': zero, 'v-': zero, 'v+': zero}, {'u': 1, 'v': 1}, variable_attrs, value_attrs)
    elif isinstance(geo, Sphere):
        thickness = geo.radius * 2
        u, v = (rel_separation * thickness * math.meshgrid(u=2, v=2)).vector
        points = geo.center + wrap([u, v, 0], geo.shape['vector'])
        if per_vertex_thickness:
            thickness = expand(thickness, spatial(points))
        return SplineSolid(points, thickness, {'u-': one, 'u+': one, 'v-': one, 'v+': one}, {'u': 1, 'v': 1}, variable_attrs, value_attrs)
    else:
        raise NotImplementedError(type(geo))


def apply_spline_bounds(spline: SplineSolid, min_thickness=1e-5):
    def rectify(points):
        p0 = points.u[0].v[0]
        dv = points.u[0].v[1] - p0
        du_ = points.u[1].v[0] - p0
        du = closest_normal_vector(du_, dv)
        du *= vec_length(du_) / vec_length(du)
        return stack([p0, p0 + dv, p0 + du, p0 + dv + du], spatial(points))
    fillet = {k: clip(v) for k, v in spline.fillet.items()}
    return replace(spline, points=rectify(spline.points), fillet=fillet, thickness=maximum(min_thickness, spline.thickness))


def transform_with_spline(points: Tensor, source: SplineSolid, target: SplineSolid):
    sdf, delta, normal, _, idx, on_skeleton, uv, unbounded_uv, tangents, normal_c, src_thickness, is_edge, is_corner = SplineSolid._closest_surface(source, points)
    is_edge = dmax(is_edge)
    # --- decompose into normal, tangent_u, 3rd axis ---
    src_tangent = vec_normalize(dmax(tangents, key=is_edge.spline.as_dual('tangents')))  # ToDo check if this is correct
    src_basis = stack([normal_c, src_tangent, cross(normal_c, src_tangent)], channel(basis='normal,tan,ortho'))
    components = src_basis.vector @ (points - on_skeleton)
    # --- transfer to target ---
    target_skeleton, target_tangents, target_normal = spline_eval(target.order, target.points, uv, ('position', 'tangents', 'normal'))
    target_tangent = vec_normalize(dmax(target_tangents, key=is_edge.spline.as_dual('tangents')))
    target_basis = stack([target_normal, target_tangent, cross(target_normal, target_tangent)], dual(basis='normal,tan,ortho'))
    thickness_change = target.thickness_at(uv) - src_thickness
    components += vec('basis', normal=thickness_change * sign(components['normal']), tan=0, ortho=0)  # add thickness difference
    # ToDo in ortho direction, thickness changes are multiplied by the fillet
    return target_skeleton + target_basis @ components
