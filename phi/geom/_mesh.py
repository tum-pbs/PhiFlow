import warnings
from numbers import Number
from typing import Dict, List, Sequence, Union, Any, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix

from phi.geom import Box
from phiml.math import to_format, is_sparse, non_channel, non_batch, batch, pack_dims
from phiml.math.extrapolation import as_extrapolation
from phiml.math.magic import slicing_dict
from ._geom import Geometry, Point
from ._graph import Graph
from .. import math
from ..math import Tensor, Shape, channel, shape, instance, dual, rename_dims, expand, spatial, wrap, sparse_tensor, stack, vec_length, tensor_like, \
    pairwise_distances, concat, Extrapolation


class Mesh(Geometry):
    """
    Unstructured mesh.
    Use `phi.geom.mesh()` or `phi.geom.mesh_from_numpy()` to construct a mesh manually or `phi.geom.load_su2()` to load one from a file.
    """

    def __init__(self,
                 vertices: Graph,
                 elements: Tensor,
                 element_rank: int,
                 boundaries: Dict[str, Dict[str, slice]],
                 center: Tensor,
                 volume: Tensor,
                 face_centers: Tensor,
                 face_normals: Tensor,
                 face_areas: Tensor,
                 face_vertices: Tensor,
                 max_cell_walk: int = None):
        """
        Args:
            vertices: Vertex positions, shape (vertices:i, vector:c)
                Vertex 0 must be at position 0.
            elements: Sparse `Tensor` listing ordered vertex indices per cell. (cells, vertices).
                The vertex count is equal to the number of elements per row.
            face_vertices: (cells, ~neighbors, face_vertices)
        """
        assert elements.dtype.kind == int, f"elements must be integer lists but got dtype {elements.dtype}"
        assert isinstance(vertices, Graph), f"vertices must be a Graph"
        self._vertices = vertices
        self._elements = elements
        self._element_rank = element_rank
        self._boundaries = boundaries
        self._center = center
        self._volume = volume
        self._face_centers = face_centers
        self._face_normals = face_normals
        self._face_areas = face_areas
        self._face_vertices = face_vertices
        cell_deltas = pairwise_distances(self.center, format=self.cell_connectivity, default=None)
        cell_distances = math.vec_length(cell_deltas)
        neighbors_dim = dual(face_areas)
        assert (cell_distances > 0).all, f"All cells must have distance > 0 but found 0 distance at {math.nonzero(cell_distances == 0)}"
        face_distances = math.vec_length(self.face_centers[self.interior_faces] - self.center)
        self._relative_face_distance = math.concat([face_distances / cell_distances, self.boundary_connectivity], neighbors_dim)
        boundary_deltas = (self.face_centers - self.center)[self.all_boundary_faces]
        assert (math.vec_length(boundary_deltas) > 0).all, f"All boundary faces must be separated from the cell centers but 0 distance at the following {channel(math.stored_indices(boundary_deltas)).item_names[0]}:\n{math.nonzero(math.vec_length(boundary_deltas) == 0):full}"
        self._neighbor_offsets = math.concat([cell_deltas, boundary_deltas], neighbors_dim)
        if max_cell_walk is None:
            max_cell_walk = 2 if instance(elements).volume > 1 else 1
        self._max_cell_walk = max_cell_walk

    def __variable_attrs__(self):
        return '_vertices', '_elements', '_center', '_volume', '_face_centers', '_face_normals', '_face_areas', '_face_vertices', '_relative_face_distance', '_neighbor_offsets'

    def __value_attrs__(self):
        return '_vertices',

    @property
    def shape(self) -> Shape:
        return shape(self._elements).non_dual & channel(self._vertices)

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
        return self.face_areas.shape

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
    def vertices(self) -> Graph:
        return self._vertices

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

    def lies_inside(self, location: Tensor) -> Tensor:
        idx = math.find_closest(self._center, location)
        for i in range(self._max_cell_walk):
            idx, leaves_mesh, is_outside, *_ = self.cell_walk_towards(location, idx, allow_exit=i == self._max_cell_walk - 1)
        return ~(leaves_mesh & is_outside)

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        idx = math.find_closest(self._center, location)
        for i in range(self._max_cell_walk):
            idx, leaves_mesh, is_outside, distances, nb_idx = self.cell_walk_towards(location, idx, allow_exit=False)
        return math.max(distances, dual)

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    @property
    def bounds(self):
        return Box(math.min(self._vertices.center, instance), math.max(self._vertices.center, instance))

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __getitem__(self, item):
        item: dict = slicing_dict(self, item)
        assert not spatial(self._elements).only(tuple(item)), f"Cannot slice vertex lists ('{spatial(self._elements)}') but got slicing dict {item}"
        assert not instance(self._vertices).only(tuple(item)), f"Slicing by vertex indices ('{instance(self._vertices)}') not supported but got slicing dict {item}"
        cells = instance(self.shape).name
        if cells in item and isinstance(item['cells'], int):
            item[cells] = slice(item[cells], item[cells] + 1)
        vertices = self._vertices[item]
        polygons = self._elements[item]
        return Mesh(vertices, polygons, self._element_rank, self._boundaries, self._center[item], self._volume[item],
                    self._face_centers[item], self._face_normals[item], self._face_areas[item], self._face_vertices[item])


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
                    boundaries: Dict[str, List[Sequence]],
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
            The elements must be listed along an instance dimension, and the vertex indices belonging to the same polygon must be listed along a spatial dimension.
        boundaries: Pass a `str` to assign one name to all boundary faces.
            For multiple boundaries, pass a `dict` mapping group names `str` to lists of faces, defined by their vertices.
            The last entry can be `None` to group all boundary faces not explicitly listed before.
            The `boundaries` `dict` maps boundary names to a list of edges (point pairs) in 2D and faces (3 or more points) in 3D (not yet supported).
        face_format: Storage format for cell connectivity, must be one of `csc`, `coo`, `csr`, `dense`.

    Returns:
        `Mesh`
    """
    assert 'vector' in channel(vertices), f"vertices must have a channel dimension called 'vector' but got {shape(vertices)}"
    assert instance(vertices), f"vertices must have an instance dimension listing all vertices of the mesh but got {shape(vertices)}"
    if spatial(polygons):  # all elements have same number of vertices
        vertex_count = spatial(polygons).size
        indices: Tensor = math.flatten(polygons)
        pointers = math.range(indices.shape['flat'] + 1, step=vertex_count)
        values = expand(1, instance(indices))
        from phiml.math._sparse import CompressedSparseMatrix
        polygons = CompressedSparseMatrix(indices, pointers, values, instance(vertices).as_dual(), instance(polygons), True)
    assert instance(vertices).as_dual() in polygons.shape, f"elements must have the instance dim of vertices {instance(vertices)} but got {shape(polygons)}"
    if vertices.vector.size == 2:
        element_rank = 2
        centers, normals, areas, boundary_slices, vertex_connectivity, face_vertices = build_faces_2d(vertices, polygons, boundaries, face_format)
    elif vertices.vector.size == 3:
        min_vertices = math.min(math.sum(polygons, instance(vertices).as_dual()))
        if min_vertices <= 4:  # assume tri or quad mesh
            element_rank = 2
            centers, normals, areas, boundary_slices, vertex_connectivity, face_vertices = build_faces_2d(vertices, polygons, boundaries, face_format)
        else:
            element_rank = 3
            raise NotImplementedError("Building 3D meshes not yet implemented. You may build them manually.")
    else:
        raise NotImplementedError(f"dim={vertices.vector.size} not supported")
    # --- Compute centers, volume ---
    vertex_count = math.sum(polygons, instance(vertices).as_dual())
    approx_center = (polygons @ vertices) / vertex_count
    normals_out = normals.vector * (centers - approx_center).vector > 0
    normals = math.where(normals_out, normals, -normals)
    vol_contributions = centers.vector * normals.vector * areas / vertices.vector.size
    volume = math.sum(vol_contributions, dual)
    cell_centers = math.sum(centers * areas, dual) / math.sum(areas, dual)
    vertices = Graph(vertices, vertex_connectivity, {})
    return Mesh(vertices, polygons, element_rank, boundary_slices, cell_centers, volume, centers, normals, areas, face_vertices)


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
    polygon_csr: csr_matrix = native_matrix(polygons, math.NUMPY)
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
    if channel(vertices).size == 2:
        normal = stack([-delta[1], delta[0]], channel(vertices))
        normal /= vec_length(normal)
    elif channel(vertices).size == 3:  # Surface mesh in 3D
        warnings.warn("Normals not yet supported for embedded 3D meshes. Using placeholder values.", RuntimeWarning, stacklevel=3)
        normal = math.random_normal(instance(delta), channel(vertices))  # ToDo
    # --- Faces ---
    dual_poly_dim = dual(neighbors=neighbor_count)
    area = sparse_tensor(indices, area, instance(polygons) & dual_poly_dim, format='coo' if face_format == 'dense' else face_format, indices_constant=True)
    normal = tensor_like(area, normal, value_order='original')
    center = tensor_like(area, center, value_order='original')
    face_centers = to_format(center, face_format)
    face_normals = to_format(normal, face_format)
    face_areas = to_format(area, face_format)
    # --- vertex-vertex connectivity ---
    vert_pairs = stack([wrap(points1 + points2 + b_points1 + b_points2, instance('edges')), wrap(points2 + points1 + b_points2 + b_points1, instance('edges'))], channel(idx=[non_channel(vertices).name, '~neighbors']))
    vertex_connectivity = sparse_tensor(vert_pairs, expand(True, instance(vert_pairs)), non_channel(vertices) & dual(neighbors=non_channel(vertices).size), can_contain_double_entries=False, indices_sorted=False, indices_constant=True)
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
