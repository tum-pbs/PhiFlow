from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Sequence, Union, Any, Tuple, Optional

import numpy as np
from phi.geom import UniformGrid, Box

from phiml.math import to_format, is_sparse, unstack, non_channel, non_batch, batch, pack_dims
from phiml.math.extrapolation import as_extrapolation
from phiml.math.magic import slicing_dict
from ._geom import Geometry, Point
from ._graph import Graph
from .. import math
from ..math import Tensor, Shape, channel, NUMPY, shape, instance, dual, rename_dims, expand, spatial, wrap, sparse_tensor, vec, stack, vec_length, tensor_like, \
    pairwise_distances, concat, Extrapolation


@dataclass
class Face:
    center: Tensor
    normal: Tensor
    area: Tensor
    # relative_distance_from_cell: Tensor
    # """Distance to primary (non-dual) cell between 0 and 1"""


class Mesh(Geometry):
    """
    Unstructured mesh.
    Use `phi.geom.mesh()` or `phi.geom.mesh_from_numpy()` to construct a mesh manually or `phi.geom.load_su2()` to load one from a file.
    """

    def __init__(self, vertices: Graph,
                 polygons: Tensor,
                 vertex_count: float or Tensor,
                 boundaries: Dict[str, Dict[str, slice]],
                 center: Tensor,
                 volume: Tensor,
                 faces: Face,
                 valid_mask: Tensor,
                 face_vertices: Tensor):
        """
        Args:
            vertices: Vertex positions, shape (vertices:i, vector:c)
                Vertex 0 must be at position 0.
            polygons: `Tensor` listing ordered vertex indices per cell.
                Must have one instance dimensions listing polygons.
                Must have one spatial dimension listing vertex indices per polygon.
                This can be a sparse or dense tensor.
                Invalid indices (index >= vertex_count) must still represent existing vertices. (or -1?)
            vertex_count: Number of vertices per polygon, shape (polygons,)
            face_vertices: (cells, ~neighbors, face_vertices)
        """
        vertex_count = expand(vertex_count, instance(polygons))
        assert polygons.dtype.kind == int, f"polygons must be integer lists but got dtype {polygons.dtype}"
        assert isinstance(vertices, Graph), f"vertices must be a Graph"
        self._vertices = vertices
        self._polygons = polygons
        self._vertex_count = vertex_count
        self._boundaries = boundaries
        self._center = center
        self._volume = volume
        self._faces = faces  # shapes (cells, ~cells+boundaries) -> int
        self._valid_mask = valid_mask
        self._face_vertices = face_vertices
        cell_deltas = pairwise_distances(self.center, format=self.cell_connectivity, default=None)
        cell_distances = math.vec_length(cell_deltas)
        face_distances = math.vec_length(self.face_centers[self.interior_faces] - self.center)
        self._relative_face_distance = math.concat([face_distances / cell_distances, self.boundary_connectivity], '~neighbors')
        boundary_deltas = (self.face_centers - self.center)[self.all_boundary_faces]
        self._neighbor_offsets = math.concat([cell_deltas, boundary_deltas], '~neighbors')
        # --- skewness ---
        # theta_e = math.PI * (vertex_count - 2) / vertex_count
        # e_face =

    def __variable_attrs__(self):
        return '_vertices', '_polygons', '_vertex_count', '_center', '_volume', '_faces', '_valid_mask', '_face_vertices', '_relative_face_distance', '_neighbor_offsets'

    def __value_attrs__(self):
        return '_vertices',

    @property
    def shape(self) -> Shape:
        return shape(self._polygons).non_spatial & channel(self._vertices)

    @property
    def cell_count(self):
        return instance(self._polygons).size

    @property
    def center(self) -> Tensor:
        return self._center

    @property
    def face_centers(self) -> Tensor:
        return self._faces.center

    @property
    def face_areas(self) -> Tensor:
        return self._faces.area

    @property
    def face_normals(self) -> Tensor:
        return self._faces.normal

    @property
    def face_shape(self) -> Shape:
        return self.face_areas.shape

    @property
    def boundary_elements(self) -> Dict[str, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        return self._boundaries

    @property
    def all_boundary_faces(self) -> Dict[str, slice]:
        return {'~neighbors': slice(instance(self).volume, None)}
    
    @property
    def interior_faces(self) -> Dict[str, slice]:
        return {'~neighbors': slice(0, instance(self).volume)}

    def pad_boundary(self, value: Tensor, widths: Dict[str, Dict[str, slice]] = None, mode: Extrapolation or Tensor or Number = 0, **kwargs) -> Tensor:
        mode = as_extrapolation(mode)
        if '~neighbors' not in value.shape:
            value = math.replace_dims(value, instance, dual('neighbors'))
        else:
            raise NotImplementedError
        if widths is None:
            widths = self.boundary_faces
        if isinstance(widths, (tuple, list)):
            if len(widths) == 0 or isinstance(widths[0], dict):  # add sliced-off slices
                pass
        dim = next(iter(next(iter(widths.values()))))
        slices = [slice(0, value.shape.get_size(dim))]
        values = [value]
        connectivity = self.connectivity
        for name, b_slice in widths.items():
            if b_slice[dim].stop - b_slice[dim].start > 0:
                slices.append(b_slice[dim])
                values.append(mode.sparse_pad_values(value, connectivity[b_slice], name, mesh=self, **kwargs))
        perm = np.argsort([s.start for s in slices])
        ordered_pieces = [values[i] for i in perm]
        return concat(ordered_pieces, dim, expand_values=True)

    @property
    def cell_connectivity(self) -> Tensor:
        """
        Returns a bool-like matrix whose non-zero entries denote connected elements.
        In meshes or grids, elements are connected if they share a face in 3D, an edge in 2D, or a vertex in 1D.

        Returns:
            `Tensor` of shape (elements, ~elements)
        """
        return self.connectivity[self.interior_faces]

    @property
    def boundary_connectivity(self) -> Tensor:
        return self.connectivity[self.all_boundary_faces]

    @property
    def connectivity(self) -> Tensor:
        if is_sparse(self._faces.area):
            return tensor_like(self._faces.area, True)
        else:
            return self._faces.area > 0

    @property
    def distance_matrix(self):
        return math.vec_length(math.pairwise_distances(self.center, edges=self.cell_connectivity, format='as edges', default=None))

    def faces_to_vertices(self, values: Tensor, reduce=sum):
        v = math.stored_values(values, invalid='keep')  # ToDo replace this once PhiML has support for dense instance dims and sparse scatter
        i = math.stored_values(self._face_vertices, invalid='keep')
        i = rename_dims(i, channel, instance)
        out_shape = non_channel(self._vertices) & shape(values).without(self.face_shape)
        return math.scatter(out_shape, i, v, mode=reduce, outside_handling='undefined')

    @property
    def relative_face_distance(self):
        """|face_center - center| / |neighbor_center - center|"""
        return self._relative_face_distance

    @property
    def neighbor_offsets(self):
        """Returns shift vector to neighbor centroids and boundary faces."""
        return self._neighbor_offsets

    @property
    def neighbor_distances(self):
        return vec_length(self._neighbor_offsets)

    @property
    def faces(self) -> 'Geometry':
        """
        Assembles information about the boundaries of the polygons that make up the surface.
        For 2D polygons, the faces are edges, for 3D polygons, the faces are planar polygons.

        Returns:
            center: Center of face connecting a pair of polygons. Shape (~polygons, polygons, vector).
                Returns 0-vectors for unconnected polygons.
            area: Area of face connecting a pair of polygons. Shape (~polygons, polygons).
                Returns 0 for unconnected polygons.
            normal: Normal vector of face connecting a pair of polygons. Shape (~polygons, polygons, vector).
                Unconnected polygons are assigned the vector 0.
                The vector points out of polygon and into ~polygon.
        """
        return Point(self.face_centers)

    @property
    def vertices(self) -> Graph:
        return self._vertices

    @property
    def polygons(self):
        return self._polygons

    @property
    def volume(self) -> Tensor:
        return self._volume

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        center = self.center
        vertex_pos = self._vertices.center[self._polygons]
        max_dist = math.max(math.vec_length(vertex_pos - center) * self._valid_mask, spatial)
        return max_dist

    def bounding_half_extent(self) -> Tensor:
        center = self.center
        vertex_pos = self._vertices.center[{instance: self._polygons}]
        max_delta = math.max(abs(vertex_pos - center) * self._valid_mask, spatial)
        return max_delta

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __getitem__(self, item):
        item: dict = slicing_dict(self, item)
        assert not spatial(self._polygons).only(tuple(item)), f"Cannot slice vertex lists ('{spatial(self._polygons)}') but got slicing dict {item}"
        assert not instance(self._vertices).only(tuple(item)), f"Slicing by vertex indices ('{instance(self._vertices)}') not supported but got slicing dict {item}"
        cells = instance(self.shape).name
        if cells in item and isinstance(item['cells'], int):
            item[cells] = slice(item[cells], item[cells] + 1)
        vertices = self._vertices[item]
        polygons = self._polygons[item]
        vertex_count = self._vertex_count[item]
        faces = Face(self._faces.center[item], self._faces.normal[item], self._faces.area[item])
        return Mesh(vertices, polygons, vertex_count, self._boundaries, self._center[item], self._volume[item], faces, self._valid_mask[item], self._face_vertices[item])


def load_su2(file_or_mesh: str, cell_dim=instance('cells'), face_format: str = 'csc') -> Mesh:
    """
    Loads an unstructured mesh from a `.su2` file.

    Args:
        file_or_mesh: Path to `.su2` file or *ezmesh* `Mesh` instance.
        cell_dim: Dimension along which to list the cells. This should be an instance dimension.
        face_format: Sparse storage format for cell connectivity.

    Returns:
        `Mesh`
    """
    if isinstance(file_or_mesh, str):
        from ezmesh import import_from_file
        mesh = import_from_file(file_or_mesh)
    else:
        mesh = file_or_mesh
    if mesh.dim == 2 and mesh.points.shape[-1] == 3:
        points = mesh.points[..., :2]
    else:
        points = mesh.points
    boundaries = {name.strip(): markers for name, markers in mesh.markers.items()}
    return mesh_from_numpy(points, mesh.elements, boundaries, cell_dim, face_format=face_format)


def mesh_from_numpy(points: Union[list, np.ndarray],
                    polygons: list,
                    boundaries: Dict[str, List[Sequence]],
                    cell_dim: Shape = instance('cells'),
                    face_format: str = 'csc') -> Mesh:
    """
    Construct an unstructured mesh from vertices.

    Args:
        points: 2D numpy array of shape (num_points, point_coord).
            The last dimension must have length 2 for 2D meshes and 3 for 3D meshes.
        polygons: List of polygons. Each polygon is defined as a sequence of point indices mapping into `points'.
            E.g. `[(0, 1, 2)]` denotes a single triangle connecting points 0, 1, and 2.
        boundaries: An unstructured mesh can have multiple boundaries, each defined by a name `str` and a list of faces, defined by their vertices.
            The `boundaries` `dict` maps boundary names to a list of edges (point pairs) in 2D and faces (3 or more points) in 3D (not yet supported).
        cell_dim: Dimension along which to list the cells. This should be an instance dimension.
        face_format: Storage format for cell connectivity, must be one of `csc`, `coo`, `csr`, `dense`.

    Returns:
        `Mesh`
    """
    cell_dim = cell_dim.with_size(len(polygons))
    points = np.asarray(points)
    try:
        elements_np = np.stack(polygons).astype(np.int32)
    except ValueError:
        vertex_count = wrap([len(e) for e in polygons], cell_dim)
        max_len = vertex_count.max
        elements_np = np.zeros((len(polygons), max_len), dtype=np.int32) - 1
        for i, element in enumerate(polygons):
            elements_np[i, :len(element)] = element
    xyz = tuple('xyz'[:points.shape[-1]])
    vertices = wrap(points, instance('vertices'), channel(vector=xyz))
    polygons = wrap(elements_np, cell_dim, spatial('vertex_index'))
    return mesh(vertices, polygons, boundaries, face_format=face_format)


def mesh(vertices: Tensor,
         polygons: Tensor,
         boundaries: Union[str, Dict[str, List[Sequence]]],
         face_format: str = 'csc'):
    """
    Create a mesh from vertex positions and vertex lists.

    Args:
        vertices: `Tensor` with one instance and one channel dimension `vector`.
        polygons: Lists of vertex indices as 2D tensor.
            The polygons must be listed along an instance dimension, and the vertex indices belonging to the same polygon must be listed along a spatial dimension.
        boundaries: Pass a `str` to assign one name to all boundary faces.
            For multiple boundaries, pass a `dict` mapping group names `str` to lists of faces, defined by their vertices.
            The last entry can be `None` to group all boundary faces not explicitly listed before.
            The `boundaries` `dict` maps boundary names to a list of edges (point pairs) in 2D and faces (3 or more points) in 3D (not yet supported).
        face_format: Storage format for cell connectivity, must be one of `csc`, `coo`, `csr`, `dense`.

    Returns:
        `Mesh`
    """
    assert 'vector' in channel(vertices), f"vertices must have a channel dimension called 'vector' but got {shape(vertices)}"
    assert instance(polygons).rank == 1, f"polygons must have exactly one instance dimension listing the polygons (cells) but got {shape(polygons)}"
    assert spatial(polygons).rank == 1, f"polygons must have exactly one spatial dimensions listing the vertices of the polygons bot got {shape(polygons)}"
    if vertices.vector.size == 2:
        faces, boundary_slices, vertex_connectivity, face_vertices = build_faces_2d(vertices, polygons, boundaries, face_format)
    elif vertices.vector.size == 3:
        raise NotImplementedError
    else:
        raise NotImplementedError(f"dim={vertices.vector.size} not supported")
    # --- Compute centers, volume ---
    valid_mask = polygons >= 0
    vertex_count = math.sum(valid_mask, spatial)
    vertex_pos = vertices.vertices[polygons]
    approx_center = math.sum(vertex_pos * valid_mask, spatial) / vertex_count
    normals_out = faces.normal.vector * (faces.center - approx_center).vector > 0
    new_normals = math.where(normals_out, faces.normal, -faces.normal)
    faces = Face(faces.center, new_normals, faces.area)
    vol_contributions = faces.center.vector * faces.normal.vector * faces.area / vertices.vector.size
    volume = math.sum(vol_contributions, dual)
    cell_centers = math.sum(faces.center * faces.area, dual) / math.sum(faces.area, dual)
    vertices = Graph(vertices, vertex_connectivity, {})
    return Mesh(vertices, polygons, vertex_count, boundary_slices, cell_centers, volume, faces, valid_mask, face_vertices)


def build_faces_2d(vertices: Tensor,
                   polygons: Tensor,
                   boundaries: Union[str, Dict[str, List[Sequence]]],
                   face_format: str):
    poly_by_face = {}  # (v1, v2) -> poly_idx
    poly1 = []
    poly2 = []
    points1 = []
    points2 = []
    # --- Find neighbor cells ---
    for poly_idx, vert_indices in enumerate(unstack(polygons, instance)):
        n_vert = int(math.sum(vert_indices >= 0))
        vert_indices = vert_indices.numpy()
        for i in range(n_vert):
            v1 = vert_indices[i]
            v2 = vert_indices[(i+1) % n_vert]
            face = (v1, v2) if v1 < v2 else (v2, v1)
            if face in poly_by_face:
                other_poly_idx = poly_by_face[face]
                del poly_by_face[face]
                poly1.append(poly_idx)
                poly2.append(other_poly_idx)
                points1.append(v1)
                points2.append(v2)
            else:
                poly_by_face[face] = poly_idx
    # --- Add boundary faces ---
    b_poly1 = []
    b_poly2 = []
    b_points1 = []
    b_points2 = []
    boundary_idx = instance(polygons).size
    boundary_slices = {}
    if not isinstance(boundaries, dict):
        boundaries = {boundaries: None}
    for boundary_name, pair_list in boundaries.items():
        boundary_start_idx = boundary_idx
        if pair_list is not None:
            for v1, v2 in pair_list:
                poly_idx = poly_by_face.get((v1, v2), None)
                if poly_idx is None:
                    poly_idx = poly_by_face.get((v2, v1), None)
                    assert poly_idx is not None, f"Boundary edge between vertices {v1} and {v2} is not connected to any cell! Either add a connected polygon or remove it from the boundary '{boundary_name}'"
                    del poly_by_face[(v2, v1)]
                    b_points1.append(v2)
                    b_points2.append(v1)
                else:
                    del poly_by_face[(v1, v2)]
                    b_points1.append(v1)
                    b_points2.append(v2)
                b_poly1.append(poly_idx)
                b_poly2.append(boundary_idx)
                boundary_idx += 1
        else:  # auto-fill rest
            for (v1, v2), poly_idx in poly_by_face.items():
                b_points1.append(v1)
                b_points2.append(v2)
                b_poly1.append(poly_idx)
                b_poly2.append(boundary_idx)
                boundary_idx += 1
            poly_by_face.clear()
        boundary_slices[boundary_name] = {'~neighbors': slice(boundary_start_idx, boundary_idx)}
    assert not poly_by_face, f"{len(poly_by_face)} edges are not marked and do not have a neighbor cell: {tuple(poly_by_face)}"
    neighbor_count = boundary_idx
    # --- wrap results as Φ-Flow tensors ---
    poly_pairs = np.asarray([poly1 + poly2 + b_poly1, poly2 + poly1 + b_poly2]).T  # include transpose of inner faces
    face_dim = instance('faces')
    indices = wrap(poly_pairs, face_dim, channel(vector=[instance(polygons).name, '~neighbors']))
    point_idx1 = wrap(points1 + points2 + b_points1, face_dim)
    point_idx2 = wrap(points2 + points1 + b_points2, face_dim)
    loc_points1 = vertices[{instance: point_idx1}]
    loc_points2 = vertices[{instance: point_idx2}]
    # --- Compute edge properties ---
    delta = loc_points2 - loc_points1
    area = vec_length(delta)
    center = (loc_points1 + loc_points2) / 2
    normal = stack([-delta[1], delta[0]], channel(vertices))
    normal /= vec_length(normal)
    # --- Faces ---
    dual_poly_dim = dual(neighbors=neighbor_count)
    area = sparse_tensor(indices, area, instance(polygons) & dual_poly_dim, format='coo' if face_format == 'dense' else face_format, indices_constant=True)
    normal = tensor_like(area, normal, value_order='original')
    center = tensor_like(area, center, value_order='original')
    faces = Face(to_format(center, face_format), to_format(normal, face_format), to_format(area, face_format))
    # --- vertex-vertex connectivity ---
    vert_pairs = stack([wrap(points1 + points2 + b_points1 + b_points2, instance('edges')), wrap(points2 + points1 + b_points2 + b_points1, instance('edges'))], channel(idx=[non_channel(vertices).name, '~neighbors']))
    vertex_connectivity = sparse_tensor(vert_pairs, expand(True, instance(vert_pairs)), non_channel(vertices) & dual(neighbors=non_channel(vertices).size), can_contain_double_entries=False, indices_sorted=False, indices_constant=True)
    # --- vertex-face connectivity ---
    vertex_pairs = stack([point_idx1, point_idx2], channel('face_vertices'))
    face_vertices = tensor_like(area, vertex_pairs, value_order='original')
    return faces, boundary_slices, vertex_connectivity, face_vertices


def build_mesh(bounds: Box = None,
               resolution=math.EMPTY_SHAPE,
               obstacles: Union[Geometry, Dict[str, Geometry]] = None,
               method='quad',
               cell_dim: Shape = instance('cells'),
               face_format: str = 'csc',
               max_squish: Optional[float] = None,
               **resolution_: Union[int, Tensor, tuple, list, Any]) -> Mesh:
    """
    Build a mesh for a given domain, respecting obstacles.

    Args:
        bounds: Bounds for uniform cells.
        resolution: Base resolution
        obstacles: Single `Geometry` or `dict` mapping boundary name to corresponding `Geometry`.
        method: Meshing algorithm. Only `quad` is currently supported.
        cell_dim: Dimension along which to list the cells. This should be an instance dimension.
        face_format: Sparse storage format for cell connectivity.
        **resolution_: For uniform grid, pass resolution as `int` and specify `bounds`.
            Or pass a sequence of floats for each dimension, specifying the vertex positions along each axis.
            This allows for variable cell stretching.

    Returns:
        `Mesh`
    """
    if obstacles is None:
        obstacles = {}
    elif isinstance(obstacles, Geometry):
        obstacles = {'obstacle': obstacles}
    assert isinstance(obstacles, dict), f"obstacles needs to be a Geometry or dict"
    if method == 'quad':
        if bounds is None:  # **resolution_ specifies points
            assert not resolution, f"When specifying vertex positions, bounds and resolution will be inferred and must not be specified."
            resolution = spatial(**{dim: non_batch(x).volume for dim, x in resolution_.items()}) - 1
            vert_pos = math.meshgrid(**resolution_)
            # centroid_x = {dim: .5 * (wrap(x[:-1]) + wrap(x[1:])) for dim, x in resolution_.items()}
            # centroids = math.meshgrid(**centroid_x)
        else:  # uniform grid from bounds, resolution
            resolution = resolution & spatial(**resolution_)
            vert_pos = math.meshgrid(resolution + 1) / resolution * bounds.size + bounds.lower
            # centroids = UniformGrid(resolution, bounds).center
        vert_pos, polygons, boundaries = build_quadrilaterals(vert_pos, resolution, obstacles)
        if max_squish is not None:
            dx = {dim: (wrap(x[1:]) - wrap(x[:-1])) for dim, x in resolution_.items()}
            regular_size = min([s.min for s in dx.values()])
            lin_vert_pos = pack_dims(vert_pos, spatial, instance('polygon'))
            corner_pos = lin_vert_pos[polygons]
            min_pos = math.min(corner_pos, '~polygon')
            max_pos = math.max(corner_pos, '~polygon')
            cell_sizes = math.min(max_pos - min_pos, 'vector')
            too_small = cell_sizes < regular_size * max_squish
            # --- remove too small cells ---
            removed = polygons[too_small]
            removed_centers = math.mean(lin_vert_pos[removed], '~polygon')
            kept_vert = removed[{'~polygon': 0}]
            vert_pos = math.scatter(lin_vert_pos, kept_vert, removed_centers)
            vertex_map = math.range(non_channel(lin_vert_pos))
            vertex_map = math.scatter(vertex_map, rename_dims(removed, '~polygon', instance('poly_list')), expand(kept_vert, instance(poly_list=4)))
            polygons = polygons[~too_small]
            polygons = vertex_map[polygons]
            boundaries = {boundary: vertex_map[edge_list] for boundary, edge_list in boundaries.items()}
            boundaries = {boundary: edge_list[edge_list[{'~vert': 'start'}] != edge_list[{'~vert': 'end'}]] for boundary, edge_list in boundaries.items()}
            # ToDo remove eges which now point to the same vertex
        def build_single_mesh(vert_pos, polygons, boundaries):
            points_np = math.reshaped_numpy(vert_pos, [..., channel])
            polygon_list = math.reshaped_numpy(polygons, [..., dual])
            boundaries = {b: edges.numpy('edges,~vert') for b, edges in boundaries.items()}
            return mesh_from_numpy(points_np, polygon_list, boundaries, cell_dim=cell_dim, face_format=face_format)
        return math.map(build_single_mesh, vert_pos, polygons, boundaries, dims=batch)


def build_quadrilaterals(vert_pos, resolution: Shape, obstacles: Dict[str, Geometry]) -> Tuple[Tensor, Tensor, dict]:
    vert_id = math.range_tensor(resolution + 1)
    # --- obstacles: mask and boundaries ---
    boundaries = {}
    full_mask = expand(False, resolution)
    for boundary, obstacle in obstacles.items():
        assert isinstance(obstacle, Geometry), f"all obstacles must be Geometry objects but got {type(obstacle)}"
        obs_mask_vert = obstacle.lies_inside(vert_pos)
        obs_mask_cell = math.convolve(~obs_mask_vert, expand(1, resolution.with_sizes(2))) == 0  # use all cells with one non-blocked vertex
        math.assert_close(False, obs_mask_cell & full_mask, msg="Obstacles must not overlap. For overlapping obstacles, use union() to assign a single boundary.")
        lo, up = math.shift(obs_mask_cell, (0, 1), padding=None)
        face_mask = lo != up
        for dim, dim_mask in dict(**face_mask.shift).items():
            face_verts = vert_id[{dim: slice(1, -1)}]
            start_vert = face_verts[{d: slice(None, -1) for d in resolution.names if d != dim}]
            end_vert = face_verts[{d: slice(1, None) for d in resolution.names if d != dim}]
            mask_indices = math.nonzero(face_mask.shift[dim], list_dim=instance('edges'))
            edges = stack([start_vert[mask_indices], end_vert[mask_indices]], dual(vert='start,end'))
            boundaries.setdefault(boundary, []).append(edges)
            # edge_list = [(s, e) for s, e, m in zip(start_vert, end_vert, dim_mask) if m]
            # boundaries.setdefault(boundary, []).extend(edge_list)
        full_mask |= obs_mask_cell
    boundaries = {boundary: concat(edge_tensors, 'edges') for boundary, edge_tensors in boundaries.items()}
    # --- outer boundaries ---
    def all_faces(ids: Tensor, edge_mask: Tensor, dim):
        assert ids.rank == 1
        mask_indices = math.nonzero(~edge_mask, list_dim=instance('edges'))
        start_vert = ids[:-1]
        end_vert = ids[1:]
        return stack([start_vert[mask_indices], end_vert[mask_indices]], dual(vert='start,end'))
        # return [(i, j) for i, j, m in zip(ids[:-1], ids[1:], edge_mask) if not m]
    for dim in resolution.names:
        boundaries[dim+'-'] = all_faces(vert_id[{dim: 0}], full_mask[{dim: 0}], dim)
        boundaries[dim+'+'] = all_faces(vert_id[{dim: -1}], full_mask[{dim: -1}], dim)
    # --- cells ---
    cell_indices = math.nonzero(~full_mask)
    if resolution.rank == 2:
        d1, d2 = resolution.names
        c1 = vert_id[{d1: slice(0, -1), d2: slice(0, -1)}]
        c2 = vert_id[{d1: slice(0, -1), d2: slice(1, None)}]
        c3 = vert_id[{d1: slice(1, None), d2: slice(1, None)}]
        c4 = vert_id[{d1: slice(1, None), d2: slice(0, -1)}]
        polygons = stack([c1, c2, c3, c4], dual('polygon'))
        polygons = polygons[cell_indices]
    else:
        raise NotImplementedError(resolution.rank)
    # --- push vertices out of obstacles ---
    ext_mask = math.pad(~full_mask, {d: (0, 1) for d in resolution.names}, False)
    has_cell = math.convolve(ext_mask, expand(1, resolution.with_sizes(2)), math.extrapolation.ZERO)  # vertices without a cell could be removed to improve memory/cache efficiency
    for obstacle in obstacles.values():
        shifted_verts = obstacle.push(vert_pos)
        vert_pos = math.where(has_cell, shifted_verts, vert_pos)
    return vert_pos, polygons, boundaries
