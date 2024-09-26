import os
import warnings
from numbers import Number
from typing import Dict, List, Sequence, Union, Any, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from phiml.math import to_format, is_sparse, non_channel, non_batch, batch, pack_dims, unstack, tensor, si2d
from phiml.math._sparse import CompactSparseTensor
from phiml.math.extrapolation import as_extrapolation
from phiml.math.magic import slicing_dict
from ._functions import plane_sgn_dist
from ._geom import Geometry, Point, scale, NoGeometry
from ._box import Box, BaseBox
from ._graph import Graph, graph
from .. import math
from ..math import Tensor, Shape, channel, shape, instance, dual, rename_dims, expand, spatial, wrap, sparse_tensor, stack, vec_length, tensor_like, \
    pairwise_distances, concat, Extrapolation


class Mesh(Geometry):
    """
    Unstructured mesh.
    Use `phi.geom.mesh()` or `phi.geom.mesh_from_numpy()` to construct a mesh manually or `phi.geom.load_su2()` to load one from a file.
    """

    def __init__(self,
                 vertices: Union[Geometry, Tensor],
                 elements: Tensor,
                 element_rank: int,
                 boundaries: Dict[str, Dict[str, slice]],
                 center: Tensor,
                 volume: Tensor,
                 normals: Optional[Tensor],
                 face_centers: Optional[Tensor],
                 face_normals: Optional[Tensor],
                 face_areas: Optional[Tensor],
                 face_vertices: Optional[Tensor],
                 vertex_normals: Optional[Tensor],
                 vertex_connectivity: Optional[Tensor],
                 element_connectivity: Optional[Tensor],
                 max_cell_walk: int = None):
        """
        Args:
            vertices: Vertex positions, shape (vertices:i, vector:c)
            elements: Sparse `Tensor` listing ordered vertex indices per cell. (cells, ~vertices).
                The vertex count is equal to the number of elements per row.
            face_vertices: (cells, ~cells, face_vertices)
        """
        assert elements.dtype.kind == int, f"elements must be integer lists but got dtype {elements.dtype}"
        assert isinstance(center, Tensor), f"center must be a Tensor"
        if not isinstance(vertices, Geometry):
            vertices = Point(vertices)
        self._vertices = vertices
        self._elements = elements
        self._element_rank = element_rank
        self._boundaries = boundaries
        self._center = center
        self._volume = volume
        self._face_centers = face_centers
        self._face_normals = face_normals
        self._face_areas = face_areas
        if self._face_areas is not None:
            assert set(face_areas.shape.names) == set((instance(elements) & dual).names), f"face_areas must have matching primal and dual dims matching elements {instance(elements)} but got {face_areas.shape}"
        self._face_vertices = face_vertices
        assert normals is None or (isinstance(normals, Tensor) and instance(center) in normals)
        self._normals = normals
        if vertex_connectivity is None and isinstance(vertices, Graph):
            self._vertex_connectivity = vertices.connectivity
        else:
            assert vertex_connectivity is None or (isinstance(vertex_connectivity, Tensor) and instance(self._vertices) in vertex_connectivity.shape), f"Illegal vertex connectivity: {vertex_connectivity}"
            self._vertex_connectivity = vertex_connectivity
        assert vertex_normals is None or (dual(vertex_normals).rank == 1 and instance(vertex_normals).rank == 0)
        self._vertex_normals = vertex_normals
        assert element_connectivity is None or isinstance(element_connectivity, Tensor), f"element_connectivity must be a Tensor"
        self._element_connectivity = element_connectivity
        if face_areas is not None or face_centers is not None or face_normals is not None:
            cell_deltas = pairwise_distances(self.center, format=self.cell_connectivity)
            cell_distances = math.vec_length(cell_deltas)
            neighbors_dim = dual(face_areas)
            assert (cell_distances > 0).all, f"All cells must have distance > 0 but found 0 distance at {math.nonzero(cell_distances == 0)}"
            face_distances = math.vec_length(self.face_centers[self.interior_faces] - self.center)
            self._relative_face_distance = math.concat([face_distances / cell_distances, self.boundary_connectivity], neighbors_dim)
            boundary_deltas = (self.face_centers - self.center)[self.all_boundary_faces]
            assert (math.vec_length(boundary_deltas) > 0).all, f"All boundary faces must be separated from the cell centers but 0 distance at the following {channel(math.stored_indices(boundary_deltas)).item_names[0]}:\n{math.nonzero(math.vec_length(boundary_deltas) == 0):full}"
            self._neighbor_offsets = math.concat([cell_deltas, boundary_deltas], neighbors_dim)
        else:
            self._relative_face_distance = None
            self._neighbor_offsets = None
        if max_cell_walk is None:
            max_cell_walk = 2 if instance(elements).volume > 1 else 1
        self._max_cell_walk = max_cell_walk

    def __variable_attrs__(self):
        return '_vertices', '_elements', '_center', '_volume', '_face_centers', '_face_normals', '_face_areas', '_face_vertices', '_normals', '_vertex_connectivity', '_vertex_normals', '_element_connectivity', '_relative_face_distance', '_neighbor_offsets'

    def __value_attrs__(self):
        return '_vertices',

    @property
    def shape(self) -> Shape:
        return shape(self._elements).non_dual & channel(self._vertices) & batch(self._vertices)

    @property
    def cell_count(self):
        return instance(self._elements).size

    @property
    def center(self) -> Tensor:
        return self._center

    @property
    def face_centers(self) -> Tensor:
        return self._face_centers

    @property
    def face_areas(self) -> Tensor:
        return self._face_areas

    @property
    def face_normals(self) -> Tensor:
        return self._face_normals

    @property
    def face_shape(self) -> Shape:
        return instance(self._elements) & dual

    @property
    def sets(self):
        return {'center': non_batch(self)-'vector', 'vertex': instance(self._vertices), '~vertex': dual(self._elements)}

    def get_points(self, set_key: str) -> Tensor:
        if set_key == 'vertex':
            return self.vertices.center
        elif set_key == '~vertex':
            return si2d(self.vertices.center)
        else:
            return Geometry.get_points(self, set_key)

    def get_boundary(self, set_key: str) -> Dict[str, Dict[str, slice]]:
        if set_key in ['vertex', '~vertex']:
            return {}
        return Geometry.get_boundary(self, set_key)

    @property
    def boundary_elements(self) -> Dict[str, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        return self._boundaries

    @property
    def _nb(self):
        return dual(self._face_areas)

    @property
    def all_boundary_faces(self) -> Dict[str, slice]:
        return {self._nb: slice(instance(self).volume, None)}
    
    @property
    def interior_faces(self) -> Dict[str, slice]:
        return {self._nb: slice(0, instance(self).volume)}

    def pad_boundary(self, value: Tensor, widths: Dict[str, Dict[str, slice]] = None, mode: Extrapolation or Tensor or Number = 0, **kwargs) -> Tensor:
        mode = as_extrapolation(mode)
        if self._nb not in value.shape:
            value = math.replace_dims(value, instance, self._nb)
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
        if self._element_connectivity is not None:
            return self._element_connectivity
        if self._face_areas is None and self._face_normals is None and self._face_centers is None:
            return None
        if is_sparse(self._face_areas):
            return tensor_like(self._face_areas, True)
        else:
            return self._face_areas > 0

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
        Assembles information about the boundaries of the elements that make up the surface.
        For 2D elements, the faces are edges, for 3D elements, the faces are planar elements.

        Returns:
            center: Center of face connecting a pair of elements. Shape (~elements, elements, vector).
                Returns 0-vectors for unconnected elements.
            area: Area of face connecting a pair of elements. Shape (~elements, elements).
                Returns 0 for unconnected elements.
            normal: Normal vector of face connecting a pair of elements. Shape (~elements, elements, vector).
                Unconnected elements are assigned the vector 0.
                The vector points out of polygon and into ~polygon.
        """
        return Point(self.face_centers)

    @property
    def vertices(self) -> Geometry:
        return self._vertices

    @property
    def vertex_connectivity(self) -> Tensor:
        return self._vertex_connectivity

    @property
    def element_connectivity(self) -> Tensor:
        return self._element_connectivity

    @property
    def vertex_graph(self) -> Graph:
        if isinstance(self._vertices, Graph):
            return self._vertices
        assert self._vertex_connectivity is not None, f"vertex_graph not available because vertex_connectivity has not been computed"
        return graph(self._vertices, self._vertex_connectivity)

    def filter_unused_vertices(self) -> 'Mesh':
        coo = math.to_format(self._elements, 'coo').numpy()
        has_element = np.asarray(coo.sum(0) > 0)[0]
        new_index = np.cumsum(has_element) - 1
        new_index_t = wrap(new_index, dual(self._elements))
        has_element = wrap(has_element, instance(self._vertices))
        has_element_d = si2d(has_element)
        vertices = self._vertices[has_element]
        v_normals = self._vertex_normals[has_element_d]
        vertex_connectivity = None
        if self._vertex_connectivity is not None:
            vertex_connectivity = math.stored_indices(self._vertex_connectivity).index.as_batch()
            vertex_connectivity = new_index_t[{dual: vertex_connectivity}].index.as_channel()
            vertex_connectivity = math.sparse_tensor(vertex_connectivity, math.stored_values(self._vertex_connectivity), non_batch(self._vertex_connectivity).with_sizes(instance(vertices).size), False)
        if isinstance(self._elements, CompactSparseTensor):
            indices = new_index_t[{dual: self._elements._indices}]
            elements = CompactSparseTensor(indices, self._elements._values, self._elements._compressed_dims.with_size(instance(vertices).volume), self._elements._indices_constant, self._elements._matrix_rank)
        else:
            filtered_coo = coo_matrix((coo.data, (coo.row, new_index)), shape=(instance(self._elements).volume, instance(vertices).volume))  # ToDo keep sparse format
            elements = wrap(filtered_coo, self._elements.shape.without_sizes())
        return Mesh(vertices, elements, self._element_rank, self._boundaries, self._center, self._volume, self._normals, self._face_centers, self._face_normals, self._face_areas, None, v_normals, vertex_connectivity, self._element_connectivity, self._max_cell_walk)

    @property
    def elements(self):
        return self._elements

    @property
    def polygons(self):
        raise NotImplementedError  # ToDo return Tensor (elements, vertex_list:spatial)

    @property
    def volume(self) -> Tensor:
        return self._volume

    @property
    def element_rank(self):
        return self._element_rank

    @property
    def normals(self) -> Tensor:
        return self._normals

    @property
    def vertex_normals(self) -> Tensor:
        return self._vertex_normals  # dual dim

    @property
    def vertex_positions(self) -> Tensor:
        return si2d(self._vertices.center)  # dual dim

    def lies_inside(self, location: Tensor) -> Tensor:
        idx = math.find_closest(self._center, location)
        for i in range(self._max_cell_walk):
            idx, leaves_mesh, is_outside, *_ = self.cell_walk_towards(location, idx, allow_exit=i == self._max_cell_walk - 1)
        return ~(leaves_mesh & is_outside)

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        if self.element_rank == 2 and self.spatial_rank == 3:
            closest_elem = math.find_closest(self._center, location)
            center = self._center[closest_elem]
            normal = self._normals[closest_elem]
            return plane_sgn_dist(center, normal, location)
        if self._center is None:
            raise NotImplementedError("Mesh.approximate_signed_distance only available when faces are built.")
        idx = math.find_closest(self._center, location)
        for i in range(self._max_cell_walk):
            idx, leaves_mesh, is_outside, distances, nb_idx = self.cell_walk_towards(location, idx, allow_exit=False)
        return math.max(distances, dual)

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.element_rank == 2 and self.spatial_rank == 3:
            closest_elem = math.find_closest(self._center, location)
            center = self._center[closest_elem]
            normal = self._normals[closest_elem]
            face_size = math.sqrt(self._volume) * 4
            size = face_size[closest_elem]
            sgn_dist = plane_sgn_dist(center, normal, location)
            delta = center - location  # this is not accurate...
            outward = math.where(abs(sgn_dist) < size, normal, math.normalize(delta))
            return sgn_dist, delta, outward, None, closest_elem
        # idx = math.find_closest(self._center, location)
        # for i in range(self._max_cell_walk):
        #     idx, leaves_mesh, is_outside, distances, nb_idx = self.cell_walk_towards(location, idx, allow_exit=False)
        # sgn_dist = math.max(distances, dual)
        # cell_normals = self.face_normals[idx]
        # normal = cell_normals[{dual: nb_idx}]
        # return sgn_dist, delta, normal, offset, face_index
        raise NotImplementedError

    def cell_walk_towards(self, location: Tensor, start_cell_idx: Tensor, allow_exit=False):
        """
        If `location` is not within the cell at index `from_cell_idx`, moves to a closer neighbor cell.

        Args:
            location: Target location as `Tensor`.
            start_cell_idx: Index of starting cell. Must be a valid cell index.
            allow_exit: If `True`, returns an invalid index for points outside the mesh, otherwise keeps the current index.

        Returns:
            index: Index of the neighbor cell or starting cell.
            leaves_mesh: Whether the walk crossed the mesh boundary. Then `index` is invalid. This is only possible if `allow_exit` is true.
            is_outside: Whether `location` was outside the cell at index `start_cell_idx`.
        """
        closest_normals = self.face_normals[start_cell_idx]
        closest_face_centers = self.face_centers[start_cell_idx]
        offsets = closest_normals.vector @ closest_face_centers.vector  # this dot product could be cashed in the mesh
        distances = closest_normals.vector @ location.vector - offsets
        is_outside = math.any(distances > 0, dual)
        nb_idx = math.argmax(distances, dual).index[0]  # cell index or boundary face index
        leaves_mesh = nb_idx >= instance(self).volume
        next_idx = math.where(is_outside & (~leaves_mesh | allow_exit), nb_idx, start_cell_idx)
        return next_idx, leaves_mesh, is_outside, distances, nb_idx

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        center = self._elements * self.center
        vert_pos = rename_dims(self._vertices.center, instance, dual)
        dist_to_vert = math.vec_length(vert_pos - center)
        max_dist = math.max(dist_to_vert, dual)
        return max_dist

    def bounding_half_extent(self) -> Tensor:
        center = self._elements * self.center
        vert_pos = rename_dims(self._vertices.center, instance, dual)
        max_delta = math.max(abs(vert_pos - center), dual)
        return max_delta

    def bounding_box(self) -> 'BaseBox':
        return self.vertices.bounding_box()

    @property
    def bounds(self):
        return Box(math.min(self._vertices.center, instance), math.max(self._vertices.center, instance))

    def at(self, center: Tensor) -> 'Mesh':
        if instance(self._elements) in center.shape:
            raise NotImplementedError("Setting Mesh positions only supported for vertices, not elements")
        if dual(self._elements) in center.shape:
            delta = rename_dims(center, dual, instance(self._vertices))
        if instance(self._vertices) in center.shape:
            vertices = self._vertices.at(center)
            return mesh(vertices, self._elements, self._boundaries, build_faces=self._face_areas is not None)
        else:
            shift = center - self.bounds.center
            return self.shifted(shift)

    def shifted(self, delta: Tensor) -> 'Mesh':
        if instance(self._elements) in delta.shape:
            raise NotImplementedError("Shifting Mesh positions only supported for vertices, not elements")
        if dual(self._elements) in delta.shape:
            delta = rename_dims(delta, dual, instance(self._vertices))
        if instance(self._vertices) in delta.shape:
            vertices = self._vertices.shifted(delta)
            return mesh(vertices, self._elements, self._boundaries, build_faces=self._face_areas is not None)
        else:  # shift everything
            vertices = self._vertices.shifted(delta)
            center = self._center + delta
            return Mesh(vertices, self._elements, self._element_rank, self._boundaries, center, self._volume, self._normals, self._face_centers, self._face_normals, self._face_areas, self._face_vertices, self._vertex_normals, self._vertex_connectivity, self._element_connectivity, self._max_cell_walk)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: float | Tensor) -> 'Geometry':
        pivot = self.bounds.center
        vertices = scale(self._vertices, factor, pivot)
        center = scale(Point(self._center), factor, pivot).center
        volume = self._volume * factor**self._element_rank if self._volume is not None else None
        face_areas = None
        return Mesh(vertices, self._elements, self._element_rank, self._boundaries, center, volume, self._normals, self._face_centers, self._face_normals, face_areas, self._face_vertices, self._vertex_normals, self._vertex_connectivity, self._element_connectivity, self._max_cell_walk)

    def __getitem__(self, item):
        item: dict = slicing_dict(self, item)
        assert not spatial(self._elements).only(tuple(item)), f"Cannot slice vertex lists ('{spatial(self._elements)}') but got slicing dict {item}"
        assert not instance(self._vertices).only(tuple(item)), f"Slicing by vertex indices ('{instance(self._vertices)}') not supported but got slicing dict {item}"
        cells = instance(self.shape).name
        if cells in item and isinstance(item[cells], int):
            item[cells] = slice(item[cells], item[cells] + 1)
        vertices = self._vertices[item]
        polygons = self._elements[item]
        s = math.slice
        return Mesh(vertices, polygons, self._element_rank, self._boundaries, self._center[item], self._volume[item], s(self._normals, item),
                    s(self._face_centers, item), s(self._face_normals, item), s(self._face_areas, item), s(self._face_vertices, item),
                    s(self._vertex_normals, item), s(self._vertex_connectivity, item), None, self._max_cell_walk)


def load_su2(file_or_mesh: str, cell_dim=instance('cells'), face_format: str = 'csc') -> Mesh:
    """
    Load an unstructured mesh from a `.su2` file.

    This requires the package `ezmesh` to be installed.

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
        assert mesh.dim == 3, f"Only 2D and 3D meshes are supported but got {mesh.dim} in {file_or_mesh}"
        points = mesh.points
    boundaries = {name.strip(): markers for name, markers in mesh.markers.items()}
    return mesh_from_numpy(points, mesh.elements, boundaries, cell_dim=cell_dim, face_format=face_format)


def load_gmsh(file: str, boundary_names: Sequence[str] = None, cell_dim=instance('cells'), face_format: str = 'csc'):
    """
    Load an unstructured mesh from a `.msh` file.

    This requires the package `meshio` to be installed.

    Args:
        file: Path to `.su2` file.
        boundary_names: Boundary identifiers corresponding to the blocks in the file. If not specified, boundaries will be numbered.
        cell_dim: Dimension along which to list the cells. This should be an instance dimension.
        face_format: Sparse storage format for cell connectivity.

    Returns:
        `Mesh`
    """
    import meshio
    from meshio import Mesh
    mesh: Mesh = meshio.read(file)
    dim = max([c.dim for c in mesh.cells])
    if dim == 2 and mesh.points.shape[-1] == 3:
        points = mesh.points[..., :2]
    else:
        assert dim == 3, f"Only 2D and 3D meshes are supported but got {dim} in {file}"
        points = mesh.points
    elements = []
    boundaries = {}
    for cell_block in mesh.cells:
        if cell_block.dim == dim:  # cells
            elements.extend(cell_block.data)
        elif cell_block.dim == dim - 1:
            # derive name from cell_block.tags if present?
            boundary = str(len(boundaries)) if boundary_names is None else boundary_names[len(boundaries)]
            boundaries[boundary] = cell_block.data
        else:
            raise AssertionError(f"Illegal cell block of type {cell_block.type} for {dim}D mesh")
    return mesh_from_numpy(points, elements, boundaries, cell_dim=cell_dim, face_format=face_format)


def mesh_from_numpy(points: Union[list, np.ndarray],
                    polygons: list,
                    boundaries: str | Dict[str, List[Sequence]] | None = None,
                    element_rank: int = None,
                    build_faces=True,
                    build_vertex_connectivity=True,
                    build_normals = True,
                    normals=None,
                    cell_dim: Shape = instance('cells'),
                    face_format: str = 'csc') -> Mesh:
    """
    Construct an unstructured mesh from vertices.

    Args:
        points: 2D numpy array of shape (num_points, point_coord).
            The last dimension must have length 2 for 2D meshes and 3 for 3D meshes.
        polygons: List of elements. Each polygon is defined as a sequence of point indices mapping into `points'.
            E.g. `[(0, 1, 2)]` denotes a single triangle connecting points 0, 1, and 2.
        boundaries: An unstructured mesh can have multiple boundaries, each defined by a name `str` and a list of faces, defined by their vertices.
            The `boundaries` `dict` maps boundary names to a list of edges (point pairs) in 2D and faces (3 or more points) in 3D (not yet supported).
        build_faces: Whether to extract face information from the given vertex, polygon and boundary information.
        build_vertex_connectivity: Whether to build a connectivity matrix for vertex-vertex connections.
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
        vertex_count = math.to_int32(wrap([len(e) for e in polygons], cell_dim))
        max_len = vertex_count.max
        elements_np = np.zeros((len(polygons), max_len), dtype=np.int32) - 1
        for i, element in enumerate(polygons):
            elements_np[i, :len(element)] = element
    xyz = tuple('xyz'[:points.shape[-1]])
    vertices = wrap(points, instance('vertices'), channel(vector=xyz))
    polygons = wrap(elements_np, cell_dim, spatial('vertex_index'))
    if normals is not None:
        normals = wrap(normals, cell_dim, channel(vector=xyz))
    return mesh(vertices, polygons, boundaries, element_rank=element_rank, build_faces=build_faces, build_vertex_connectivity=build_vertex_connectivity, build_normals=build_normals, face_format=face_format, normals=normals)


@math.broadcast(dims=batch)
def mesh(vertices: Geometry | Tensor,
         elements: Tensor,
         boundaries: str | Dict[str, List[Sequence]] | None = None,
         element_rank: int = None,
         build_faces=True,
         build_vertex_connectivity=True,
         build_element_connectivity=True,
         build_normals=True,
         normals=None,
         face_format: str = 'csc'):
    """
    Create a mesh from vertex positions and vertex lists.

    Args:
        vertices: `Tensor` with one instance and one channel dimension `vector`.
        elements: Lists of vertex indices as 2D tensor.
            The elements must be listed along an instance dimension, and the vertex indices belonging to the same polygon must be listed along a spatial dimension.
        boundaries: Pass a `str` to assign one name to all boundary faces.
            For multiple boundaries, pass a `dict` mapping group names `str` to lists of faces, defined by their vertices.
            The last entry can be `None` to group all boundary faces not explicitly listed before.
            The `boundaries` `dict` maps boundary names to a list of edges (point pairs) in 2D and faces (3 or more points) in 3D (not yet supported).
        build_faces: Whether to extract face information from the given vertex, polygon and boundary information.
        build_vertex_connectivity: Whether to build a connectivity matrix for vertex-vertex connections.
        face_format: Storage format for cell connectivity, must be one of `csc`, `coo`, `csr`, `dense`.

    Returns:
        `Mesh`
    """
    assert 'vector' in channel(vertices), f"vertices must have a channel dimension called 'vector' but got {shape(vertices)}"
    assert instance(vertices), f"vertices must have an instance dimension listing all vertices of the mesh but got {shape(vertices)}"
    if not isinstance(vertices, Geometry):
        vertices = Point(vertices)
    if build_faces:
        assert boundaries is not None, f"When build_faces=True, boundaries must be specified."
    if spatial(elements):  # all elements have same number of vertices
        indices: Tensor = rename_dims(elements, spatial, instance(vertices).as_dual())
        values = expand(1, non_batch(indices))
        elements = CompactSparseTensor(indices, values, instance(vertices).as_dual(), instance(elements))
    assert instance(vertices).as_dual() in elements.shape, f"elements must have the instance dim of vertices {instance(vertices)} but got {shape(elements)}"
    if element_rank is None:
        if vertices.vector.size == 2:
            element_rank = 2
        elif vertices.vector.size == 3:
            min_vertices = math.sum(elements, instance(vertices).as_dual()).min
            element_rank = 2 if min_vertices <= 4 else 3  # assume tri or quad mesh
        else:
            raise ValueError(vertices.vector.size)
    vertex_count = math.sum(elements, instance(vertices).as_dual())
    approx_center = (elements @ vertices.center) / vertex_count
    # --- build faces ---
    if build_faces:
        if element_rank == 2:
            centers, normals, areas, boundary_slices, vertex_connectivity, face_vertices = build_faces_2d(vertices.center, elements, boundaries, face_format)
        else:
            raise NotImplementedError("Building faces currently only supported for 2D elements. Set build_faces=False to construct mesh")
        normals_out = normals.vector * (centers - approx_center).vector > 0
        normals = math.where(normals_out, normals, -normals)
        vol_contributions = centers.vector * normals.vector * areas / vertices.vector.size
        volume = math.sum(vol_contributions, dual)
        cell_centers = math.sum(centers * areas, dual) / math.sum(areas, dual)
        return Mesh(vertices, elements, element_rank, boundary_slices, cell_centers, volume, None, centers, normals, areas, face_vertices, None, vertex_connectivity, None)
    else:
        vertex_connectivity = None
        if build_vertex_connectivity:
            coo = math.to_format(elements, 'coo').numpy()
            connected_points = coo.T @ coo
            if not np.all(connected_points.sum(axis=1) > 0):
                warnings.warn("some vertices have no element connection at all", RuntimeWarning)
            connected_points.data = np.ones_like(connected_points.data)
            vertex_connectivity = wrap(connected_points, instance(vertices), dual(elements))
        element_connectivity = None
        if build_element_connectivity:
            coo = math.to_format(elements, 'coo').numpy()
            connected_elements = coo @ coo.T
            connected_elements.data = np.ones_like(connected_elements.data)
            element_connectivity = wrap(connected_elements, instance(elements), instance(elements).as_dual())
        volume = None
        if isinstance(elements, CompactSparseTensor) and element_rank == 2:
            if instance(vertices).volume > 0:
                A, B, C, *_ = unstack(vertices.center[elements._indices], dual)
                cross_area = math.vec_length(math.cross_product(B - A, C - A))
                fac = {3: 0.5, 4: 1}[dual(elements._indices).size]  # tri, quad, ...
                volume = fac * cross_area
            else:
                volume = math.zeros(instance(vertices))
        if normals is None and build_normals:
            normals = extrinsic_normals(vertices.center, elements)
        v_normals = None
        if build_normals:
            v_normals = vertex_normals(elements, normals)
        return Mesh(vertices, elements, element_rank, {}, approx_center, volume, normals, None, None, None, None, v_normals, vertex_connectivity, element_connectivity)


def build_faces_2d(vertices: Tensor,
                   polygons: Tensor,
                   boundaries: Union[str, Dict[str, List[Sequence]]],
                   face_format: str):
    poly_by_face = {}  # (v1, v2) -> poly_idx
    poly1 = []
    poly2 = []
    points1 = []
    points2 = []
    from phiml.math._sparse import native_matrix
    polygon_csr: csr_matrix = native_matrix(math.to_format(polygons, 'csr'), math.NUMPY)
    # --- Find neighbor cells ---
    for poly_idx in range(instance(polygons).size):
        vert_indices = polygon_csr.indices[polygon_csr.indptr[poly_idx]:polygon_csr.indptr[poly_idx+1]]
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
    # --- Add boundary faces ---
    b_poly1 = []
    b_poly2 = []
    b_points1 = []
    b_points2 = []
    boundary_idx = instance(polygons).size
    boundary_slices = {}
    nb_dim = instance(polygons).as_dual().name
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
        boundary_slices[boundary_name] = {nb_dim: slice(boundary_start_idx, boundary_idx)}
    assert not poly_by_face, f"{len(poly_by_face)} edges are not marked and do not have a neighbor cell: {tuple(poly_by_face)}"
    neighbor_count = boundary_idx
    # --- wrap results as Φ-Flow tensors ---
    poly_pairs = np.asarray([poly1 + poly2 + b_poly1, poly2 + poly1 + b_poly2]).T  # include transpose of inner faces
    face_dim = instance('faces')
    indices = wrap(poly_pairs, face_dim, channel(vector=[instance(polygons).name, nb_dim]))
    point_idx1 = wrap(points1 + points2 + b_points1, face_dim)
    point_idx2 = wrap(points2 + points1 + b_points2, face_dim)
    loc_points1 = vertices[{instance: point_idx1}]
    loc_points2 = vertices[{instance: point_idx2}]
    # --- Compute edge properties ---
    delta = loc_points2 - loc_points1
    area = vec_length(delta)
    center = (loc_points1 + loc_points2) / 2
    if channel(vertices).size == 2:
        normal = stack([-delta[1], delta[0]], channel(vertices))
        normal /= vec_length(normal)
    elif channel(vertices).size == 3:  # Surface mesh in 3D
        warnings.warn("Normals not yet supported for embedded 3D meshes. Using placeholder values.", RuntimeWarning, stacklevel=3)
        normal = math.random_normal(instance(delta), channel(vertices))  # ToDo
    # --- Faces ---
    dual_poly_dim = dual(**{nb_dim: neighbor_count})
    area = sparse_tensor(indices, area, instance(polygons) & dual_poly_dim, format='coo' if face_format == 'dense' else face_format, indices_constant=True)
    normal = tensor_like(area, normal, value_order='original')
    center = tensor_like(area, center, value_order='original')
    face_centers = to_format(center, face_format)
    face_normals = to_format(normal, face_format)
    face_areas = to_format(area, face_format)
    # --- vertex-vertex connectivity ---
    vert_pairs = stack([wrap(points1 + points2 + b_points1 + b_points2, instance('edges')), wrap(points2 + points1 + b_points2 + b_points1, instance('edges'))], channel(idx=[non_channel(vertices).name, nb_dim]))
    vertex_connectivity = sparse_tensor(vert_pairs, expand(True, instance(vert_pairs)), non_channel(vertices) & dual(**{nb_dim: non_channel(vertices).size}), can_contain_double_entries=False, indices_sorted=False, indices_constant=True)
    # --- vertex-face connectivity ---
    vertex_pairs = stack([point_idx1, point_idx2], channel('face_vertices'))
    face_vertices = tensor_like(area, vertex_pairs, value_order='original')
    return face_centers, face_normals, face_areas, boundary_slices, vertex_connectivity, face_vertices


def build_mesh(bounds: Box = None,
               resolution=math.EMPTY_SHAPE,
               obstacles: Union[Geometry, Dict[str, Geometry]] = None,
               method='quad',
               cell_dim: Shape = instance('cells'),
               face_format: str = 'csc',
               max_squish: Optional[float] = .5,
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
        max_squish: Smallest allowed cell size compared to the smallest regular cell.
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
            bounds = Box(**{dim: (x[0], x[-1]) for dim, x in resolution_.items()})
            # centroid_x = {dim: .5 * (wrap(x[:-1]) + wrap(x[1:])) for dim, x in resolution_.items()}
            # centroids = math.meshgrid(**centroid_x)
        else:  # uniform grid from bounds, resolution
            resolution = resolution & spatial(**resolution_)
            vert_pos = math.meshgrid(resolution + 1) / resolution * bounds.size + bounds.lower
            # centroids = UniformGrid(resolution, bounds).center
        dx = bounds.size / resolution
        regular_size = math.min(dx, channel)
        vert_pos, polygons, boundaries = build_quadrilaterals(vert_pos, resolution, obstacles, bounds, regular_size * max_squish)
        if max_squish is not None:
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
            # ToDo remove edges which now point to the same vertex
        def build_single_mesh(vert_pos, polygons, boundaries):
            points_np = math.reshaped_numpy(vert_pos, [..., channel])
            polygon_list = math.reshaped_numpy(polygons, [..., dual])
            boundaries = {b: edges.numpy('edges,~vert') for b, edges in boundaries.items()}
            return mesh_from_numpy(points_np, polygon_list, boundaries, cell_dim=cell_dim, face_format=face_format)
        return math.map(build_single_mesh, vert_pos, polygons, boundaries, dims=batch)


def build_quadrilaterals(vert_pos, resolution: Shape, obstacles: Dict[str, Geometry], bounds: Box, min_size) -> Tuple[Tensor, Tensor, dict]:
    vert_id = math.range_tensor(resolution + 1)
    # --- obstacles: mask and boundaries ---
    boundaries = {}
    full_mask = expand(False, resolution)
    for boundary, obstacle in obstacles.items():
        assert isinstance(obstacle, Geometry), f"all obstacles must be Geometry objects but got {type(obstacle)}"
        active_mask_vert = obstacle.approximate_signed_distance(vert_pos) > min_size
        obs_mask_cell = math.convolve(active_mask_vert, expand(1, resolution.with_sizes(2))) == 0  # use all cells with one non-blocked vertex
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
    vert_pos = bounds.push(vert_pos, outward=False)
    return vert_pos, polygons, boundaries


def tri_points(mesh: Mesh):
    corners = mesh.vertices.center[mesh._elements._indices]
    assert dual(corners).size == 3, f"signed distance currently only supports triangles"
    return unstack(corners, dual)


def extrinsic_normals(vertices: Tensor, elements: Tensor):
    corners = vertices[elements._indices]
    assert dual(corners).size == 3, f"signed distance currently only supports triangles"
    A, B, C = unstack(corners, dual)
    return math.vec_normalize(math.cross_product(B-A, C-A))


def vertex_normals(elements: Tensor, face_normals: Tensor):
    v_normals = math.mean(elements * face_normals, instance)  # (~vertices,vector)
    return math.vec_normalize(v_normals)


def face_curvature(mesh: Mesh):
    v_normals = mesh.elements * si2d(mesh.vertex_normals)
    # v_offsets = mesh.elements * si2d(mesh.vertices.center) - mesh.center

    corners = mesh.vertices.center[mesh.elements._indices]
    assert dual(corners).size == 3, f"signed distance currently only supports triangles"
    A, B, C = unstack(corners.vector.as_dual(), dual(corners))
    e1, e2, e3 = B-A, C-B, A-C
    n1, n2, n3 = unstack(v_normals._values, dual)
    dn1, dn2, dn3 = n2-n1, n3-n2, n1-n3
    curvature_tensor = .5 / mesh.volume * (e1 * dn1 + e2 * dn2 + e3 * dn3)
    scalar_curvature = math.sum([curvature_tensor[{'vector': d, '~vector': d}] for d in mesh.vector.item_names], '0')
    return curvature_tensor, scalar_curvature
    # vec_curvature = math.max(v_normals, dual) - math.min(v_normals, dual)  # positive / negative


def save_tri_mesh(file: str, mesh: Mesh):
    v = math.reshaped_numpy(mesh.vertices.center, [instance, 'vector'])
    if isinstance(mesh._elements, CompactSparseTensor):
        f = math.reshaped_numpy(mesh._elements._indices, [instance, dual])
    else:
        raise NotImplementedError
    print(f"Saving triangle mesh with {v.shape[0]} vertices and {f.shape[0]} faces to {file}")
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez(file, vertices=v, faces=f, f_dim=instance(mesh).name, vertex_dim=instance(mesh.vertices).name, vector=mesh.vector.item_names)


def load_tri_mesh(file: str, convert=False) -> Mesh:
    data = np.load(file)
    f_dim = instance(str(data['f_dim']))
    vertex_dim = instance(str(data['vertex_dim']))
    vector = channel(vector=[str(d) for d in data['vector']])
    faces = tensor(data['faces'], f_dim, spatial('vertex_list'), convert=convert)
    vertices = tensor(data['vertices'], vertex_dim, vector, convert=convert)
    return mesh(vertices, faces, build_faces=False, build_vertex_connectivity=True, build_normals=True)


def decimate_tri_mesh(mesh: Mesh, factor=.1, target_max=10_000,):
    if isinstance(mesh, NoGeometry):
        return mesh
    if instance(mesh).volume == 0:
        return mesh
    import pyfqmr
    mesh_simplifier = pyfqmr.Simplify()
    vertices = math.reshaped_numpy(mesh.vertices.center, [instance, 'vector'])
    faces = math.reshaped_numpy(mesh.elements._indices, [instance, dual])
    target_count = min(target_max, int(round(instance(mesh).volume * factor)))
    mesh_simplifier.setMesh(vertices, faces)
    mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=7, preserve_border=False)
    vertices, faces, normals = mesh_simplifier.getMesh()
    return mesh_from_numpy(vertices, faces, normals=normals, build_faces=False, build_vertex_connectivity=mesh._vertex_connectivity is not None, cell_dim=instance(mesh))
