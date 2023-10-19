from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Dict, List, Sequence, Union

import numpy as np

from phiml.math.extrapolation import as_extrapolation
from ._geom import Geometry, Point
from .. import math
from ..math import Tensor, Shape, channel, NUMPY, shape, instance, dual, rename_dims, expand, spatial, pack_dims, wrap, sparse_tensor, vec, stack, vec_length, tensor_like, \
    pairwise_distances, concat, Extrapolation


@dataclass
class Face:
    center: Tensor
    normal: Tensor
    area: Tensor
    # relative_distance_from_cell: Tensor
    # """Distance to primary (non-dual) cell between 0 and 1"""


class UnstructuredMesh(Geometry):

    def __init__(self, vertices: Tensor,
                 polygons: Tensor,
                 vertex_count: float or Tensor,
                 boundaries: Dict[str, Dict[str, slice]],
                 center: Tensor,
                 volume: Tensor,
                 faces: Face,
                 valid_mask: Tensor):
        """
        Args:
            vertices: Vertex positions, shape (vertices:i, vector:c)
                Vertex 0 must be at position 0.
            polygons: `Tensor` containing vertex indices, Shape (polygons:i, vertex_index:s).
                This can be a sparse or dense tensor.
                Invalid indices (index >= vertex_count) must still represent existing vertices. (or -1?)
            vertex_count: Number of vertices per polygon, shape (polygons,)
        """
        assert instance(vertices).rank == 1, f"vertices must have exactly one instance dimension but got {shape(vertices)}"
        vertices = rename_dims(vertices, instance, 'vertices')
        assert 'vector' in channel(vertices) and channel(vertices).get_item_names('vector') is not None, "vertices must have a 'vector' dim listing the physical dimensions"
        vertex_count = expand(vertex_count, instance(polygons))
        assert 'vertex_index' in spatial(polygons), f"polygons must have exactly one spatial dimension called 'vertex_index' but got {shape(polygons)}"
        assert polygons.dtype.kind == int, f"polygons must be integer lists but got dtype {polygons.dtype}"
        self._vertices = vertices
        self._polygons = polygons
        self._vertex_count = vertex_count
        self._boundaries = boundaries
        self._center = center
        self._volume = volume
        self._faces = faces  # shapes (cells, ~cells+boundaries) -> int
        self._valid_mask = valid_mask
        cell_deltas = pairwise_distances(self.center, format=self.cell_connectivity, default=None)
        cell_distances = math.vec_length(cell_deltas)
        face_distances = math.vec_length(self.face_centers[self.interior_faces] - self.center)
        self._relative_face_distance = math.concat([face_distances / cell_distances, math.tensor_like(self.boundary_connectivity, 1)], '~neighbors')
        boundary_deltas = (self.face_centers - self.center)[self.all_boundary_faces]
        self._neighbor_offsets = math.concat([cell_deltas, boundary_deltas], '~neighbors')
        # --- skewness ---
        # theta_e = math.PI * (vertex_count - 2) / vertex_count
        # e_face =

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
                values.append(mode.sparse_pad_values(value, connectivity[b_slice], name, **kwargs))
        perm = np.argsort([s.start for s in slices])
        ordered_pieces = [values[i] for i in perm]
        return concat(ordered_pieces, dim, expand_values=True)

    # def edges(self, bidirectional=False) -> Tuple[Tensor, Tensor]:
    #     last_vertex = expand(self._polygons.vertex_index[self._vertex_count-1], spatial(vertex_index=1))
    #     vertex1 = math.concat([last_vertex, self._polygons.vertex_index[:-1]], 'vertex_index')
    #     vertex2 = self._polygons
    #     indices = math.vec(from_vertex=vertex1, to_vertex=vertex2)
    #     indices = pack_dims(indices, ['vertex_index', instance(self._polygons)], instance('edges'))
    #     vdim = list(instance(self._vertices).names)
    #     indices = rename_dims(indices, 'vector', channel(vector=['~'+s for s in vdim] + vdim))
    #     valid = self._valid_mask
    #     values = pack_dims(valid, 'vertex_index', instance('edges'))  # ToDo pack
    #     dense_shape = dual(vertices=self._vertices.vertices.size) & instance(self._vertices)
    #     edges = sparse_tensor(indices, values, dense_shape, can_contain_double_entries=True, indices_sorted=False)
    #     # ToDo remove zero values
    #     if bidirectional:
    #         pass  # ToDo add transpose to make matrix symmetric, then remove doubles
    #     # ToDo remove doubles
    #     return self._vertices, edges

    @property
    def cell_connectivity(self) -> Tensor:
        """
        Returns a bool-like matrix whose non-zero entries denote connected elements.
        In meshes or grids, elements are connected if they share a face in 3D, an edge in 2D, or a vertex in 1D.

        Returns:
            `Tensor` of shape (elements, ~elements)
        """
        return tensor_like(self._faces.area, True)[self.interior_faces]

    @property
    def boundary_connectivity(self) -> Tensor:
        return tensor_like(self._faces.area, True)[self.all_boundary_faces]

    @property
    def connectivity(self) -> Tensor:
        return tensor_like(self._faces.area, True)

    @property
    def distance_matrix(self):
        return math.vec_length(math.pairwise_distances(self.center, edges=self.cell_connectivity, format='as edges', default=None))

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
    def vertices(self):
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

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        center = self.center
        vertex_pos = self._vertices[self._polygons]
        max_dist = math.max(math.vec_length(vertex_pos - center) * self._valid_mask, 'vertex_index')
        return max_dist

    def bounding_half_extent(self) -> Tensor:
        center = self.center
        vertex_pos = self._vertices[self._polygons]
        max_delta = math.max(abs(vertex_pos - center) * self._valid_mask, 'vertex_index')
        return max_delta

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __hash__(self):
        return hash((self._vertices, self._polygons))

    def __getitem__(self, item):
        assert 'vertices' not in item, "Cannot slice UnstructuredMesh along 'vertices'"
        assert 'vertex_index' not in item, "Cannot slice UnstructuredMesh along 'vertex_index'"
        vertices = self._vertices[item]
        polygons = self._polygons[item]
        vertex_count = self._vertex_count[item]
        faces = Face(self._faces.center[item], self._faces.normal[item], self._faces.area[item])
        return UnstructuredMesh(vertices, polygons, vertex_count, self._boundaries, self._center[item], self._volume[item], faces, self._valid_mask[item])


def load_su2(file_or_mesh: str, cell_dim=instance('cells'), face_format: str = 'csc') -> UnstructuredMesh:
    """
    Loads an unstructured mesh from a `.su2` file.

    Args:
        file_or_mesh: Path to `.su2` file or *ezmesh* `Mesh` instance.
        cell_dim: Dimension along which to list the cells. This should be an instance dimension.
        face_format: Sparse storage format for cell connectivity.

    Returns:
        `UnstructuredMesh`
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
                    face_format: str = 'csc') -> UnstructuredMesh:
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
        face_format: Sparse storage format for cell connectivity.

    Returns:
        `UnstructuredMesh`
    """
    cell_dim = cell_dim.with_size(len(polygons))
    points = np.asarray(points)
    dim = points.shape[-1]
    if dim == 2:
        faces, boundary_slices = build_faces_2d(points, polygons, boundaries, channel(vector='x,y'), cell_dim, face_format)
        vertices = vec(x=points[:, 0].tolist(), y=points[:, 1].tolist())
    elif dim == 3:
        # vertices = vec(x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist())
        raise NotImplementedError
    else:
        raise NotImplementedError(f"dim={dim} not supported")
    try:
        elements_np = np.stack(polygons).astype(np.int32)
        vertex_count = elements_np.shape[-1]
    except ValueError:
        vertex_count = wrap([len(e) for e in polygons], cell_dim)
        max_len = vertex_count.max
        elements_np = np.zeros((len(polygons), max_len), dtype=np.int32)
        for i, element in enumerate(polygons):
            elements_np[i, :len(element)] = element
    polygons = wrap(elements_np, cell_dim, spatial('vertex_index'))
    # --- Compute centers, volume ---
    max_index = polygons.vertex_index.size
    with NUMPY:
        valid_mask = math.range(spatial(vertex_index=max_index)) < vertex_count
    vertex_pos = vertices.sequence[polygons]
    cell_centers = math.sum(vertex_pos * valid_mask, 'vertex_index') / vertex_count
    normals_out = faces.normal.vector * (faces.center - cell_centers).vector > 0
    new_normals = math.where(normals_out, faces.normal, -faces.normal)
    faces = Face(faces.center, new_normals, faces.area)
    vol_contributions = faces.center.vector * faces.normal.vector * faces.area / dim
    volume = math.sum(vol_contributions, dual(faces.area))
    return UnstructuredMesh(vertices, polygons, vertex_count, boundary_slices, cell_centers, volume, faces, valid_mask)


def build_faces_2d(points: np.ndarray,
                   polygons: list,
                   boundaries: Dict[str, List[Sequence]],
                   vector_dim: Shape,
                   poly_dim: Shape,
                   face_format: str):
    assert points.shape[-1] == 2, f"Only 2D faces implemented"
    poly_by_face = {}
    poly1 = []
    poly2 = []
    points1 = []
    points2 = []
    # --- Find neighbor cells ---
    for poly_idx, vert_indices in enumerate(polygons):
        n_vert = len(vert_indices)
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
    # --- Add transpose ---
    poly1, poly2 = poly1 + poly2, poly2 + poly1
    points1, points2 = points1 + points2, points2 + points1
    # --- Add boundary faces ---
    # dual_poly_dim = dual(**poly_dim.untyped_dict).with_size(len(polygons) + sum([len(b) for bs in boundaries.values() for b in bs]))
    dual_poly_dim = dual(neighbors=len(polygons) + sum([len(b) for b in boundaries.values()]))
    boundary_idx = len(polygons)
    boundary_slices = {}
    for boundary_name, pair_list in boundaries.items():
        boundary_start_idx = boundary_idx
        for pair in pair_list:
            v1, v2 = pair
            poly_idx = poly_by_face.get((v1, v2), None)
            if poly_idx is None:
                poly_idx = poly_by_face[(v2, v1)]
                del poly_by_face[(v2, v1)]
                points1.append(v2)
                points2.append(v1)
            else:
                del poly_by_face[(v1, v2)]
                points1.append(v1)
                points2.append(v2)
            poly1.append(poly_idx)
            poly2.append(boundary_idx)
            boundary_idx += 1
        boundary_slices[boundary_name] = {dual_poly_dim.name: slice(boundary_start_idx, boundary_idx)}
    assert not poly_by_face, f"{len(poly_by_face)} edges are not marked and do not have a neighbor cell."
    # --- wrap results as Φ-Flow tensors ---
    poly_pairs = np.asarray([poly1, poly2]).T
    face_dim = instance('faces')
    indices = wrap(poly_pairs, face_dim, channel(vector=[poly_dim.name, dual_poly_dim.name]))
    points1 = wrap(points[points1, :], face_dim, vector_dim)
    points2 = wrap(points[points2, :], face_dim, vector_dim)
    # --- Compute edge properties ---
    delta = points2 - points1
    area = vec_length(delta)
    center = (points1 + points2) / 2
    normal = stack([-delta[1], delta[0]], vector_dim)
    normal /= vec_length(normal)
    # --- Create sparse tensors ---
    area = sparse_tensor(indices, area, poly_dim & dual_poly_dim, format=face_format, default=None)
    normal = tensor_like(area, normal, value_order='original')
    center = tensor_like(area, center, value_order='original')
    return Face(center, normal, area), boundary_slices
