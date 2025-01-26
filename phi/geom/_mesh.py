import os
from dataclasses import dataclass
from functools import cached_property, lru_cache
from numbers import Number
from typing import Dict, List, Sequence, Union, Any, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from phiml import math
from phiml.math import to_format, is_sparse, non_channel, non_batch, batch, pack_dims, unstack, tensor, si2d, non_dual, nonzero, stored_indices, stored_values, scatter, \
    find_closest, sqrt, where, vec_normalize, argmax, broadcast, zeros, EMPTY_SHAPE, meshgrid, mean, reshaped_numpy, range_tensor, convolve, \
    assert_close, shift, pad, extrapolation, sum as sum_, dim_mask, math, Tensor, Shape, channel, shape, instance, dual, rename_dims, expand, spatial, wrap, sparse_tensor, \
    stack, tensor_like, pairwise_distances, concat, Extrapolation, dsum, reshaped_tensor, dmean
from phiml.dataclasses import getitem, replace
from phiml.math._sparse import CompactSparseTensor
from phiml.math.extrapolation import as_extrapolation, PERIODIC
from phiml.math.magic import slicing_dict

from ._geom import Geometry, Point, NoGeometry
from ._box import Box, BaseBox, bounding_box
from ._functions import plane_sgn_dist, cross, vec_length
from ._graph import Graph, graph
from ._transform import scale


@dataclass(frozen=True)
class Mesh(Geometry):
    """
    Unstructured mesh, consisting of vertices and elements.
    
    Use `phi.geom.mesh()` or `phi.geom.mesh_from_numpy()` to construct a mesh manually or `phi.geom.load_su2()` to load one from a file.
    """

    vertices: Geometry
    """ Vertices are represented by a `Geometry` instance with an instance dim. """
    elements: Tensor
    """ elements: Sparse `Tensor` listing ordered vertex indices per element (solid or surface element, depending on `element_rank`).
    Must have one instance dim listing the elements and the corresponding dual dim to `vertices`.
    The vertex count of an element is equal to the number of elements in that row (i.e. summing the dual dim). """
    element_rank: int
    """The spatial rank of the elements. Solid elements have the same as the ambient space, faces one less."""
    boundaries: Dict[str, Dict[str, slice]]
    """Slices to retrieve boundary face values."""
    periodic: Sequence[str]
    """List of axis names that are periodic. Periodic boundaries must be named as axis- and axis+. For example `['x']` will connect the boundaries x- and x+."""
    face_format: str = 'csc'
    """Sparse matrix format for storing quantities that depend on a pair of neighboring elements, e.g. `face_area`, `face_normal`, `face_center`."""
    max_cell_walk: int = None
    """ Maximum number of steps to walk along the element connectivity in order to find a cell, e.g. for sampling at an arbitrary point."""

    variable_attrs: Tuple[str, ...] = ('vertices',)  # PhiML keyword
    value_attrs: Tuple[str, ...] = ()  # PhiML keyword

    def __post_init__(self):
        if spatial(self.elements):
            assert self.elements.dtype.kind == int, f"elements listing vertices must be integer lists but got dtype {self.elements.dtype}"
        else:
            assert self.elements.dtype.kind == bool, f"element matrices must be of type bool but got {self.elements.dtype}"

    @cached_property
    def shape(self) -> Shape:
        return non_dual(self.elements) & channel(self.vertices) & batch(self.vertices)

    @cached_property
    def cell_count(self):
        return instance(self.elements).size

    @cached_property
    def center(self) -> Tensor:
        if self.element_rank == self.spatial_rank:  # Compute volumetric center from faces
            return sum_(self.face_centers * self.face_areas, dual) / sum_(self.face_areas, dual)
        else:  # approximate center from vertices
            return self._vertex_mean

    @cached_property
    def _vertex_mean(self):
        """Mean vertex location per element."""
        vertex_count = sum_(self.elements, instance(self.vertices).as_dual())
        return (self.elements @ self.vertices.center) / vertex_count

    @cached_property
    def face_centers(self) -> Tensor:
        return self._faces['center']

    @property
    def face_areas(self) -> Tensor:
        return self._faces['area']

    @cached_property
    def face_normals(self) -> Tensor:
        if self.element_rank == self.spatial_rank:  # this cannot depend on element centers because that depends on the normals.
            normals = self._faces['normal']
            face_centers = self._faces['center']
            normals_out = normals.vector * (face_centers - self._vertex_mean).vector > 0
            normals = where(normals_out, normals, -normals)
            return normals
        raise NotImplementedError

    @cached_property
    def _faces(self) -> Dict[str, Any]:
        centers, normals, areas, boundary_slices = build_faces(self.vertices.center, self.elements, self.boundaries, self.element_rank, self.periodic, self._vertex_mean, self.face_format)
        return {
            'center': centers,
            'normal': normals,
            'area': areas,
            'boundary_slices': boundary_slices,
        }

    def _build_faces(self):
        return build_faces(self.vertices.center, self.elements, self.boundaries, self.element_rank, self.periodic, self._vertex_mean, self.face_format)

    @property
    def face_shape(self) -> Shape:
        if not self.boundary_faces:
            return instance(self) & dual
        dual_len = max([next(iter(sl.values())).stop for sl in self.boundary_faces.values()])
        dim = instance(self)
        return dim.as_dual().with_size(dual_len) + dim

    @property
    def sets(self):
        return {
            'center': non_batch(self)-'vector',
            'vertex': instance(self.vertices),
            '~vertex': dual(self.elements)
        }

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
        if self.boundaries is None:
            return {}
        return self._faces['boundary_slices']

    @property
    def all_boundary_faces(self) -> Dict[str, slice]:
        if self.face_shape.dual.size == self.elements.shape.instance.size:
            return {}
        return {self.face_shape.dual.name: slice(instance(self).volume, None)}

    @property
    def interior_faces(self) -> Dict[str, slice]:
        return {self.face_shape.dual.name: slice(0, instance(self).volume)}

    def pad_boundary(self, value: Tensor, widths: Dict[str, Dict[str, slice]] = None, mode: Extrapolation or Tensor or Number = 0, **kwargs) -> Tensor:
        mode = as_extrapolation(mode)
        if self.face_shape.dual.name not in value.shape:
            value = rename_dims(value, instance, self.face_shape.dual.without_sizes())
        else:
            raise NotImplementedError
        if widths is None:
            widths = self.boundary_faces
        if isinstance(widths, dict) and len(widths) == 0:
            return value
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

    @cached_property
    def cell_connectivity(self) -> Tensor:
        """
        Returns a bool-like matrix whose non-zero entries denote connected elements.
        In meshes or grids, elements are connected if they share a face in 3D, an edge in 2D, or a vertex in 1D.

        Returns:
            `Tensor` of shape (elements, ~elements)
        """
        return self.connectivity[self.interior_faces]

    @cached_property
    def boundary_connectivity(self) -> Tensor:
        if not self.all_boundary_faces:
            dual_dim = instance(self).as_dual().with_size(0)
            return zeros(instance(self) & dual_dim)
        return self.connectivity[self.all_boundary_faces]

    @cached_property
    def distance_matrix(self):
        return vec_length(pairwise_distances(self.center, edges=self.cell_connectivity, format='as edges', default=None))

    def faces_to_vertices(self, values: Tensor, reduce=sum):
        v = stored_values(values, invalid='keep')  # ToDo replace this once PhiML has support for dense instance dims and sparse scatter
        i = stored_values(self.face_vertices, invalid='keep')
        i = rename_dims(i, channel, instance)
        out_shape = non_channel(self.vertices) & shape(values).without(self.face_shape)
        return scatter(out_shape, i, v, mode=reduce, outside_handling='undefined')

    @cached_property
    def _cell_deltas(self):
        bounds = bounding_box(self.vertices)
        periodic = {dim[:-len('[::-1]')] if dim.endswith('[::-1]') else dim: dim.endswith('[::-1]') for dim in self.periodic}
        is_periodic = dim_mask(self.vector.item_names, tuple(periodic))
        return pairwise_distances(self.center, format=self.cell_connectivity, periodic=is_periodic, domain=(bounds.lower, bounds.upper))

    @cached_property
    def relative_face_distance(self):
        """|face_center - center| / |neighbor_center - center|"""
        cell_distances = vec_length(self._cell_deltas)
        assert (cell_distances > 0).all, f"All cells must have distance > 0 but found 0 distance at {nonzero(cell_distances == 0)}"
        face_distances = vec_length(self.face_centers[self.interior_faces] - self.center)
        return concat([face_distances / cell_distances, self.boundary_connectivity], self.face_shape.dual)

    @cached_property
    def neighbor_offsets(self):
        """Returns shift vector to neighbor centroids and boundary faces."""
        if not self.all_boundary_faces:
            return self._cell_deltas
        boundary_deltas = (self.face_centers - self.center)[self.all_boundary_faces]
        assert (vec_length(boundary_deltas) > 0).all, f"All boundary faces must be separated from the cell centers but 0 distance at the following {channel(stored_indices(boundary_deltas)).item_names[0]}:\n{nonzero(vec_length(boundary_deltas) == 0):full}"
        return concat([self._cell_deltas, boundary_deltas], self.face_shape.dual)

    @cached_property
    def neighbor_distances(self):
        return vec_length(self.neighbor_offsets)

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
    def connectivity(self) -> Tensor:
        return self.element_connectivity

    @cached_property
    def element_connectivity(self) -> Tensor:
        """Neighbor element connectivity, excluding diagonal."""
        if self.element_rank == self.spatial_rank:
            if is_sparse(self.face_areas):
                return tensor_like(self.face_areas, True)
            else:
                return self.face_areas > 0
        else:  # fallback with no boundaries
            coo = to_format(self.elements, 'coo').numpy()
            connected_elements = coo @ coo.T
            connected_elements.setdiag(0)
            connected_elements.eliminate_zeros()
            connected_elements.data = np.ones_like(connected_elements.data)
            element_connectivity = wrap(connected_elements, instance(self.elements), instance(self.elements).as_dual())
            return element_connectivity

    @cached_property
    def vertex_connectivity(self) -> Tensor:
        if isinstance(self.vertices, Graph):
            return self.vertices.connectivity
        elif self.element_rank <= 2:
            def single_vertex_connectivity(elements: Tensor):
                indices = stored_indices(elements).index[dual(elements).name]
                idx1 = indices.numpy()
                v_count = sum_(elements, dual).numpy()
                ptr_end = np.cumsum(v_count)
                roll = np.arange(idx1.size) + 1
                roll[ptr_end-1] = ptr_end - v_count
                idx2 = idx1[roll]
                v_conn = coo_matrix((np.ones(idx1.size, dtype=bool), (idx1, idx2)), shape=(dual(elements).size,)*2).tocsr()
                return wrap(v_conn, dual(elements).as_instance(), dual(elements))
            return math.map(single_vertex_connectivity, self.elements, dims=batch)
        raise NotImplementedError

    @cached_property
    def vertex_graph(self) -> Graph:
        return self.vertices if isinstance(self.vertices, Graph) else graph(self.vertices, self.vertex_connectivity)

    def filter_unused_vertices(self) -> 'Mesh':
        coo = to_format(self.elements, 'coo').numpy()
        has_element = np.asarray(coo.sum(0) > 0)[0]
        new_index = np.cumsum(has_element) - 1
        new_index_t = wrap(new_index, dual(self.elements))
        has_element = wrap(has_element, instance(self.vertices))
        has_element_d = si2d(has_element)
        vertices = self.vertices[has_element]
        v_normals = self.vertex_normals[has_element_d]
        vertex_connectivity = None
        # if self._vertex_connectivity is not None:
        #     vertex_connectivity = stored_indices(self._vertex_connectivity).index.as_batch()
        #     vertex_connectivity = new_index_t[{dual: vertex_connectivity}].index.as_channel()
        #     vertex_connectivity = sparse_tensor(vertex_connectivity, stored_values(self._vertex_connectivity), non_batch(self._vertex_connectivity).with_sizes(instance(vertices).size), False)
        if isinstance(self.elements, CompactSparseTensor):
            indices = new_index_t[{dual: self.elements._indices}]
            elements = CompactSparseTensor(indices, self.elements._values, self.elements._compressed_dims.with_size(instance(vertices).volume), self.elements._indices_constant, self.elements._matrix_rank)
        else:
            filtered_coo = coo_matrix((coo.data, (coo.row, new_index)), shape=(instance(self.elements).volume, instance(vertices).volume))  # ToDo keep sparse format
            elements = wrap(filtered_coo, self.elements.shape.without_sizes())
        return replace(self, vertices=vertices, elements=elements)

    @cached_property
    def volume(self) -> Tensor:
        if self.element_rank == 2:
            if instance(self.elements).volume > 0:
                three_vertices = nonzero(self.elements, 3, list_dims=dual)
                v1, v2, v3 = unstack(self.vertices.center[{instance: three_vertices}], dual)
                cross_area = vec_length(cross(v2-v1, v3-v1))
                vertex_count = math.sum(self.elements, dual)
                fac = where(vertex_count == 3, 0.5, 1)  # tri, quad, ...
                return fac * cross_area
            else:
                return zeros(instance(self.vertices))  # empty mesh
        elif self.element_rank == self.spatial_rank:
            vol_contributions = (self.face_centers.vector @ self.face_normals.vector) * self.face_areas
            return sum_(vol_contributions, dual) / self.spatial_rank
        raise NotImplementedError


    @property
    def normals(self) -> Tensor:
        """Extrinsic element normal space. This is a 0D vector for solid elements and 1D for surface elements."""
        if self.element_rank == 2:
            three_vertices = nonzero(self.elements, 3, list_dims=dual)
            v1, v2, v3 = unstack(self.vertices.center[{instance: three_vertices}], dual)
            return vec_normalize(cross(v2 - v1, v3 - v1))
        raise NotImplementedError

    @property
    def vertex_normals(self) -> Tensor:
        v_normals = mean(self.elements * self.normals, instance)  # (~vertices,vector)
        return vec_normalize(v_normals)

    @property
    def vertex_positions(self) -> Tensor:
        """Lists the vertex centers along the corresponding dual dim to `self.vertices.center`."""
        return si2d(self.vertices.center)

    def lies_inside(self, location: Tensor) -> Tensor:
        idx = find_closest(self.center, location)
        for i in range(self.max_cell_walk):
            idx, leaves_mesh, is_outside, *_ = self.cell_walk_towards(location, idx, allow_exit=i == self.max_cell_walk - 1)
        return ~(leaves_mesh & is_outside)

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        if self.element_rank == 2 and self.spatial_rank == 3:
            closest_elem = find_closest(self.center, location)
            center = self.center[closest_elem]
            normal = self.normals[closest_elem]
            return plane_sgn_dist(center, normal, location)
        idx = find_closest(self.center, location)
        for i in range(self.max_cell_walk):
            idx, leaves_mesh, is_outside, distances, nb_idx = self.cell_walk_towards(location, idx, allow_exit=False)
        return math.max(distances, dual)

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.element_rank == 2 and self.spatial_rank == 3:
            closest_elem = find_closest(self.center, location)
            center = self.center[closest_elem]
            normal = self.normals[closest_elem]
            face_size = sqrt(self.volume) * 4
            size = face_size[closest_elem]
            sgn_dist = plane_sgn_dist(center, normal, location)
            delta = center - location  # this is not accurate...
            outward = where(abs(sgn_dist) < size, normal, vec_normalize(delta))
            return sgn_dist, delta, outward, None, closest_elem
        # idx = find_closest(self.center, location)
        # for i in range(self.max_cell_walk):
        #     idx, leaves_mesh, is_outside, distances, nb_idx = self.cell_walk_towards(location, idx, allow_exit=False)
        # sgn_dist = max(distances, dual)
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
        nb_idx = argmax(distances, dual).index[0]  # cell index or boundary face index
        leaves_mesh = nb_idx >= instance(self).volume
        next_idx = where(is_outside & (~leaves_mesh | allow_exit), nb_idx, start_cell_idx)
        return next_idx, leaves_mesh, is_outside, distances, nb_idx

    def sample_uniform(self, *shape: Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        center = self.elements * self.center
        vert_pos = rename_dims(self.vertices.center, instance, dual)
        dist_to_vert = vec_length(vert_pos - center)
        max_dist = math.max(dist_to_vert, dual)
        return max_dist

    def bounding_half_extent(self) -> Tensor:
        center = self.elements * self.center
        vert_pos = rename_dims(self.vertices.center, instance, dual)
        max_delta = math.max(abs(vert_pos - center), dual)
        return max_delta

    def bounding_box(self) -> 'BaseBox':
        return self.vertices.bounding_box()

    @property
    def bounds(self):
        return Box(math.min(self.vertices.center, instance), math.max(self.vertices.center, instance))

    def at(self, center: Tensor) -> 'Mesh':
        if instance(self.elements) in center.shape:
            raise NotImplementedError("Setting Mesh positions only supported for vertices, not elements")
        if dual(self.elements) in center.shape:
            center = rename_dims(center, dual, instance(self.vertices))
        if instance(self.vertices) in center.shape:
            vertices = self.vertices.at(center)
            return replace(self, vertices=vertices)
        else:
            return self.shifted(center - self.bounds.center)

    def shifted(self, delta: Tensor) -> 'Mesh':
        if instance(self.elements) in delta.shape:
            raise NotImplementedError("Shifting Mesh positions only supported for vertices, not elements")
        if dual(self.elements) in delta.shape:
            delta = rename_dims(delta, dual, instance(self.vertices))
        if instance(self.vertices) in delta.shape:
            vertices = self.vertices.shifted(delta)
            return replace(self, vertices=vertices)
        else:  # shift everything
            # ToDo transfer cached properties
            # copy: center+delta, normals, volume, face_centers+delta, face_areas, face_normals, vertex_normals, vertex_connectivity, element_connectivity
            vertices = self.vertices.shifted(delta)
            return replace(self, vertices=vertices)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        pivot = self.bounds.center
        vertices = scale(self.vertices, factor, pivot)
        # center = scale(Point(self.center), factor, pivot).center
        # volume = self.volume * factor**self.element_rank if self.volume is not None else None
        return replace(self, vertices=vertices)

    def __getitem__(self, item):
        item: dict = slicing_dict(self, item)
        assert not dual(self.elements).only(tuple(item)), f"Cannot slice vertex lists ('{spatial(self.elements)}') but got slicing dict {item}"
        return getitem(self, item, keepdims=[(self.shape.instance - instance(self.vertices)).name, 'vector'])

    def __repr__(self):
        return Geometry.__repr__(self)


@broadcast
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


@broadcast
def load_gmsh(file: str, boundary_names: Sequence[str] = None, periodic: str = None, cell_dim=instance('cells'), face_format: str = 'csc'):
    """
    Load an unstructured mesh from a `.msh` file.

    This requires the package `meshio` to be installed.

    Args:
        file: Path to `.su2` file.
        boundary_names: Boundary identifiers corresponding to the blocks in the file. If not specified, boundaries will be numbered.
        periodic:
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
    return mesh_from_numpy(points, elements, boundaries, periodic=periodic, cell_dim=cell_dim, face_format=face_format)


@broadcast
def load_stl(file: str, face_dim=instance('faces')) -> Mesh:
    """
    Load a triangle `Mesh` from an STL file.

    Args:
        file: File path to `.stl` file.
        face_dim: Instance dim along which to list the triangles.

    Returns:
        `Mesh` with `spatial_rank=3` and `element_rank=2`.
    """
    import trimesh
    mesh = trimesh.load(file)
    if isinstance(mesh, trimesh.Scene):  # STL contains multiple parts -> merge
        vertices = []
        v_count = 0
        faces = []
        for geometry in mesh.geometry.values():
            assert isinstance(geometry, trimesh.Trimesh)
            vertices.append(geometry.vertices)
            faces.append(geometry.faces + v_count)
            v_count += geometry.vertices.shape[0]
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
    else:
        assert isinstance(mesh, trimesh.Trimesh), f"Unexpected content of STL: {mesh}"
        vertices, faces = mesh.vertices, mesh.faces
    return mesh_from_numpy(vertices, faces, None, 2, None, face_dim)
    # import stl  # this only loads the first part of multi-part STL files
    # model = stl.mesh.Mesh.from_file(file, calculate_normals=False, )
    # points = np.reshape(model.points, (-1, 3))
    # vertices, indices = np.unique(points, axis=0, return_inverse=True)
    # indices = np.reshape(indices, (-1, 3))
    # mesh = mesh_from_numpy(vertices, indices, element_rank=2, cell_dim=face_dim)
    # return mesh


def mesh_from_numpy(points: Sequence[Sequence],
                    polygons: Sequence[Sequence],
                    boundaries: Union[str, Dict[str, List[Sequence]], None] = None,
                    element_rank: int = None,
                    periodic: str = None,
                    cell_dim: Shape = instance('cells'),
                    face_format: str = 'csc',
                    axes=('x', 'y', 'z')) -> Mesh:
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
    xyz = tuple(axes[:points.shape[-1]])
    vertices = wrap(points, instance('vertices'), channel(vector=xyz))
    if len(polygons) == 0:
        elements = math.ones(cell_dim, instance(vertices).as_dual(), dtype=bool)
        return mesh(vertices, elements, boundaries, element_rank, periodic, face_format)
    try:  # if all elements have the same vertex count, we stack them
        elements_np = np.stack(polygons).astype(np.int32)
        elements = wrap(elements_np, cell_dim, spatial('vertex_index'))
    except ValueError:
        indices = np.concatenate(polygons)
        vertex_count = np.asarray([len(e) for e in polygons])
        ptr = np.pad(np.cumsum(vertex_count), (1, 0))
        mat = csr_matrix((np.ones(indices.shape, dtype=bool), indices, ptr), shape=(len(polygons), len(points)))
        elements = wrap(mat, cell_dim, instance(vertices).as_dual())
    return mesh(vertices, elements, boundaries, element_rank, periodic, face_format=face_format)


@broadcast(dims=batch)
def mesh(vertices: Union[Geometry, Tensor],
         elements: Tensor,
         boundaries: Union[str, Dict[str, List[Sequence]], None] = None,
         element_rank: int = None,
         periodic: str = None,
         face_format: str = 'csc',
         max_cell_walk: int = None):
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
        face_format: Storage format for cell connectivity, must be one of `csc`, `coo`, `csr`, `dense`.

    Returns:
        `Mesh`
    """
    assert 'vector' in channel(vertices), f"vertices must have a channel dimension called 'vector' but got {shape(vertices)}"
    assert instance(vertices), f"vertices must have an instance dimension listing all vertices of the mesh but got {shape(vertices)}"
    if not isinstance(vertices, Geometry):
        vertices = Point(vertices)
    if spatial(elements):  # all elements have same number of vertices
        indices: Tensor = rename_dims(elements, spatial, instance(vertices).as_dual())
        values = expand(True, non_batch(indices))
        elements = CompactSparseTensor(indices, values, instance(vertices).as_dual(), True)
    assert instance(vertices).as_dual() in elements.shape, f"elements must have the instance dim of vertices {instance(vertices)} but got {shape(elements)}"
    if element_rank is None:
        if vertices.vector.size == 2:
            element_rank = 2
        elif vertices.vector.size == 3:
            min_vertices = sum_(elements, instance(vertices).as_dual()).min
            element_rank = 2 if min_vertices <= 4 else 3  # assume tri or quad mesh
        else:
            raise ValueError(vertices.vector.size)
    if max_cell_walk is None:
        max_cell_walk = 2 if instance(elements).volume > 1 else 1
    # --- build faces ---
    periodic_dims = []
    if periodic is not None:
        periodic_dims = [s.strip() for s in periodic.split(',') if s.strip()]
        periodic_base = [p[:-len('[::-1]')] if p.endswith('[::-1]') else p for p in periodic_dims]
        assert all(p in vertices.vector.item_names for p in periodic_base), f"Periodic boundaries must be named after axes, e.g. {vertices.vector.item_names} but got {periodic}"
        for base in periodic_base:
            assert base+'+' in boundaries and base+'-' in boundaries, f"Missing boundaries for periodicity '{base}'. Make sure '{base}+' and '{base}-' are keys in boundaries dict, got {tuple(boundaries)}"
    return Mesh(vertices, elements, element_rank, boundaries, periodic_dims, face_format=face_format, max_cell_walk=max_cell_walk)


def build_faces(vertices: Tensor,  # (vertices:i, vector)
                elements: Tensor,  # (elements:i, ~vertices)
                boundaries: Dict[str, Sequence],  # vertex pairs
                element_rank: int,
                periodic: Sequence[str],  # periodic dim names
                vertex_mean: Tensor,
                face_format: str):
    """
    Given a list of vertices, elements and boundary edges, computes the element connectivity matrix  and corresponding edge properties.

    Args:
        vertices: `Tensor` representing list (instance) of vectors (channel)
        elements: Sparse matrix listing all elements (instance). Each entry represents a vertex (dual) belonging to an element.
        boundaries: Named sequences of edges (vertex pairs).
        element_rank: Spatial rank of the elements (currently only 2 is supported)
        periodic: Which dims are periodic.
        vertex_mean: Mean vertex position for each element.
        face_format: Sparse matrix format to use for the element-element matrices.
    """
    n_v = instance(vertices).size
    n_e = instance(elements).size
    # --- Periodic: map vertices of boundary+ to the corresponding vertex in boundary- ---
    vertex_id = np.arange(instance(vertices).size)
    periodic = {dim[:-len('[::-1]')] if dim.endswith('[::-1]') else dim: dim.endswith('[::-1]') for dim in periodic}
    for dim, is_flipped in periodic.items():
        vertex_id[np.concatenate(boundaries[dim+'+'])] = vertex_id[np.concatenate(boundaries[dim+'-'])[::-1] if is_flipped else np.concatenate(boundaries[dim+'-'])]
    is_periodic = dim_mask(vertices.vector.item_names, tuple(periodic))
    # --- element-facet and facet-vertex matrix. A facet describes a single oriented face of an element, i.e. shared faces get two entries. ---
    v_count = dsum(elements).numpy()  # number of vertices per element
    v1 = stored_indices(elements).index[dual(elements).name].numpy()
    if element_rank == 2:  # edges are the lines between neighbor vertices in the vertex lists + the edge last-to-first
        v1 = vertex_id[v1]
        n_f = v1.size  # total number of facets (excluding boundaries)
        ptr = np.cumsum(v_count)
        roll = np.arange(v1.size) + 1
        roll[ptr - 1] = ptr - v_count
        v12 = np.stack([v1, v1[roll]], -1).flatten()
        f_idx = np.arange(v1.size, dtype=v1.dtype)
        f_idx2 = f_idx.repeat(2)
        f_v = coo_matrix((np.ones(n_f*2, np.int32), (f_idx2, v12)), shape=(n_f, n_v))  # facet-vertex matrix
        e_idx = np.arange(instance(elements).size).repeat(v_count)
        e_f = coo_matrix((np.ones(n_f, bool), (e_idx, f_idx)), shape=(n_e, n_f))  # element-facet matrix
        # --- Compute facet properties: center, normal, area ---
        f_v_pos = vertices[reshaped_tensor(v12, [instance('facets') + dual(pair=2)], convert=False)]  # vertex positions of every (inner) facet
        if periodic:  # map v_pos: closest to cell_center
            cell_center = vertex_mean[wrap(e_idx, 'facets:i')]
            bounds = bounding_box(vertices)
            delta = PERIODIC.shortest_distance(cell_center - bounds.lower, f_v_pos - bounds.lower, bounds.size)
            f_v_pos = where(is_periodic, cell_center + delta, f_v_pos)
        f_center = dmean(f_v_pos)
        edge_dir = f_v_pos.pair.dual[1] - f_v_pos.pair.dual[0]
        area = vec_length(edge_dir)
        normal = vec_normalize(stack([-edge_dir[1], edge_dir[0]], channel(edge_dir)))
    elif element_rank == 3:
        v3d, c3d = element_types_3d()
        n_v_per_f = [c3d[v] for v in v_count]
        n_f_per_e = [len(v) for v in n_v_per_f]
        n_fv_per_e = [sum(v) for v in n_v_per_f]
        n_v_per_f = np.concatenate(n_v_per_f)
        n_f = sum(n_f_per_e)
        f_ptr = np.pad(np.cumsum(n_v_per_f), (1, 0))
        v_idx0 = np.concatenate([v3d[v] for v in v_count])  # here vertex indices start at 0 for each element
        v_idx = v1[v_idx0 + np.pad(np.cumsum(n_f_per_e), (1, 0))[:-1].repeat(n_fv_per_e)]
        f_v = csr_matrix((np.ones(v_idx.size, np.int32), v_idx, f_ptr), shape=(n_f, n_v))
        f_idx = np.arange(n_f)
        e_ptr = np.pad(np.cumsum(n_f_per_e), (1, 0))
        e_f = csr_matrix((np. ones(n_f, bool), f_idx, e_ptr), shape=(n_e, n_f))
        # --- Compute facet properties: center, normal, area ---
        facet_vertices = wrap(f_v, 'facets:i', instance(vertices).as_dual())
        f_v_pos = facet_vertices * vertices.Ti
        if periodic:  # map v_pos: closest to cell_center
            e_idx = np.arange(n_e).repeat(n_f_per_e)
            cell_center = vertex_mean[wrap(e_idx, 'facets:i')]
            bounds = bounding_box(vertices)
            delta = PERIODIC.shortest_distance(cell_center - bounds.lower, f_v_pos - bounds.lower, bounds.size)
            f_v_pos = where(is_periodic, cell_center + delta, f_v_pos)
        f_center = dmean(f_v_pos)
        fv123 = wrap(v_idx[f_ptr[:-1] + np.arange(3)[:, None]], 'v:s=(v1,v2,v3),facets:i')
        fv_pos = vertices[fv123]
        cross_prod = cross(fv_pos.v['v2']-fv_pos.v['v1'], fv_pos.v['v3']-fv_pos.v['v1'])
        area_fac = np.where(n_v_per_f == 3, 0.5, 1)
        area = vec_length(cross_prod) * area_fac
        normal = vec_normalize(cross_prod)
    else:
        raise ValueError(f"element_rank must be 2 or 3 but got {element_rank}")
    # --- Add virtual boundary elements to f_v for non-periodic boundaries ---
    boundary_slices = {}
    e_end, f_end = e_f.shape
    b_idx_f, b_idx_v = [[i] for i in f_v.nonzero()]
    for bnd_key, bnd_vertices in boundaries.items():
        if bnd_key[:-1] in periodic:
            continue
        bv_count = np.asarray([len(vs) for vs in bnd_vertices])
        bv_idx = np.concatenate(bnd_vertices)
        f_idx = np.arange(len(bnd_vertices)).repeat(bv_count) + f_end
        b_idx_f.append(f_idx)
        b_idx_v.append(bv_idx)
        boundary_slices[bnd_key] = {instance(elements).as_dual().name: slice(e_end, e_end+len(bnd_vertices))}
        f_end += len(bnd_vertices)
        e_end += len(bnd_vertices)
    b_idx_f = np.concatenate(b_idx_f)
    b_idx_v = vertex_id[np.concatenate(b_idx_v)]
    f_v_b = coo_matrix((np.ones(b_idx_f.size, bool), (b_idx_f, b_idx_v)), shape=(f_end, n_v))
    # --- Add virtual boundary facets to e_f ---
    e_f_be = np.concatenate([e_f.nonzero()[0], np.arange(n_e, e_end)])
    e_f_bf = np.arange(e_f_be.size)  # every face assigned to exactly one element. Identical to np.concatenate([e_f.col, np.arange(n_f, f_end)])
    e_f_b = coo_matrix((np.ones(e_f_bf.size, bool), (e_f_be, e_f_bf)), shape=(e_end, f_end))
    # --- Compute connectivity and return element-pair facet properties ---
    f_f: csr_matrix = f_v @ f_v_b.T >= element_rank  # symmetric
    f_f.setdiag(0)
    f_f.eliminate_zeros()
    assert np.all((f_f > 0).sum(1) == 1), f"Each facet should have one backside but got {(f_f > 0).sum(1)}"
    f_f.data = f_f.nonzero()[0] + 1
    e_e = e_f @ f_f @ e_f_b.T  # stores the outgoing facet_index+1 for each element pair
    shared_f_idx = wrap(e_e, instance(elements).without_sizes() & dual) - 1
    shared_f_idx = to_format(shared_f_idx, face_format)
    return f_center[shared_f_idx], normal[shared_f_idx], area[shared_f_idx], boundary_slices


def build_mesh(bounds: Box = None,
               resolution=EMPTY_SHAPE,
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
            vert_pos = meshgrid(**resolution_)
            bounds = Box(**{dim: (x[0], x[-1]) for dim, x in resolution_.items()})
            # centroid_x = {dim: .5 * (wrap(x[:-1]) + wrap(x[1:])) for dim, x in resolution_.items()}
            # centroids = meshgrid(**centroid_x)
        else:  # uniform grid from bounds, resolution
            resolution = resolution & spatial(**resolution_)
            vert_pos = meshgrid(resolution + 1) / resolution * bounds.size + bounds.lower
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
            removed_centers = mean(lin_vert_pos[removed], '~polygon')
            kept_vert = removed[{'~polygon': 0}]
            vert_pos = scatter(lin_vert_pos, kept_vert, removed_centers)
            vertex_map = math.range(non_channel(lin_vert_pos))
            vertex_map = scatter(vertex_map, rename_dims(removed, '~polygon', instance('poly_list')), expand(kept_vert, instance(poly_list=4)))
            polygons = polygons[~too_small]
            polygons = vertex_map[polygons]
            boundaries = {boundary: vertex_map[edge_list] for boundary, edge_list in boundaries.items()}
            boundaries = {boundary: edge_list[edge_list[{'~vert': 'start'}] != edge_list[{'~vert': 'end'}]] for boundary, edge_list in boundaries.items()}
            # ToDo remove edges which now point to the same vertex
        def build_single_mesh(vert_pos, polygons, boundaries):
            points_np = reshaped_numpy(vert_pos, [..., channel])
            polygon_list = reshaped_numpy(polygons, [..., dual])
            boundaries = {b: edges.numpy('edges,~vert') for b, edges in boundaries.items()}
            return mesh_from_numpy(points_np, polygon_list, boundaries, cell_dim=cell_dim, face_format=face_format)
        return math.map(build_single_mesh, vert_pos, polygons, boundaries, dims=batch)


def build_quadrilaterals(vert_pos, resolution: Shape, obstacles: Dict[str, Geometry], bounds: Box, min_size) -> Tuple[Tensor, Tensor, dict]:
    vert_id = range_tensor(resolution + 1)
    # --- obstacles: mask and boundaries ---
    boundaries = {}
    full_mask = expand(False, resolution)
    for boundary, obstacle in obstacles.items():
        assert isinstance(obstacle, Geometry), f"all obstacles must be Geometry objects but got {type(obstacle)}"
        active_mask_vert = obstacle.approximate_signed_distance(vert_pos) > min_size
        obs_mask_cell = convolve(active_mask_vert, expand(1, resolution.with_sizes(2))) == 0  # use all cells with one non-blocked vertex
        assert_close(False, obs_mask_cell & full_mask, msg="Obstacles must not overlap. For overlapping obstacles, use union() to assign a single boundary.")
        lo, up = shift(obs_mask_cell, (0, 1), padding=None)
        face_mask = lo != up
        for dim, dim_mask in dict(**face_mask.shift).items():
            face_verts = vert_id[{dim: slice(1, -1)}]
            start_vert = face_verts[{d: slice(None, -1) for d in resolution.names if d != dim}]
            end_vert = face_verts[{d: slice(1, None) for d in resolution.names if d != dim}]
            mask_indices = nonzero(face_mask.shift[dim], list_dim=instance('edges'))
            edges = stack([start_vert[mask_indices], end_vert[mask_indices]], dual(vert='start,end'))
            boundaries.setdefault(boundary, []).append(edges)
            # edge_list = [(s, e) for s, e, m in zip(start_vert, end_vert, dim_mask) if m]
            # boundaries.setdefault(boundary, []).extend(edge_list)
        full_mask |= obs_mask_cell
    boundaries = {boundary: concat(edge_tensors, 'edges') for boundary, edge_tensors in boundaries.items()}
    # --- outer boundaries ---
    def all_faces(ids: Tensor, edge_mask: Tensor, dim):
        assert ids.rank == 1
        mask_indices = nonzero(~edge_mask, list_dim=instance('edges'))
        start_vert = ids[:-1]
        end_vert = ids[1:]
        return stack([start_vert[mask_indices], end_vert[mask_indices]], dual(vert='start,end'))
        # return [(i, j) for i, j, m in zip(ids[:-1], ids[1:], edge_mask) if not m]
    for dim in resolution.names:
        boundaries[dim+'-'] = all_faces(vert_id[{dim: 0}], full_mask[{dim: 0}], dim)
        boundaries[dim+'+'] = all_faces(vert_id[{dim: -1}], full_mask[{dim: -1}], dim)
    # --- cells ---
    cell_indices = nonzero(~full_mask)
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
    ext_mask = pad(~full_mask, {d: (0, 1) for d in resolution.names}, False)
    has_cell = convolve(ext_mask, expand(1, resolution.with_sizes(2)), extrapolation.ZERO)  # vertices without a cell could be removed to improve memory/cache efficiency
    for obstacle in obstacles.values():
        shifted_verts = obstacle.push(vert_pos)
        vert_pos = where(has_cell, shifted_verts, vert_pos)
    vert_pos = bounds.push(vert_pos, outward=False)
    return vert_pos, polygons, boundaries


def tri_points(mesh: Mesh):
    corners = mesh.vertices.center[mesh.elements._indices]
    assert dual(corners).size == 3, f"signed distance currently only supports triangles"
    return unstack(corners, dual)



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
    scalar_curvature = sum_([curvature_tensor[{'vector': d, '~vector': d}] for d in mesh.vector.item_names], '0')
    return curvature_tensor, scalar_curvature
    # vec_curvature = math.max(v_normals, dual) - math.min(v_normals, dual)  # positive / negative


def save_tri_mesh(file: str, mesh: Mesh, **extra_data):
    v = reshaped_numpy(mesh.vertices.center, [instance, 'vector'])
    if isinstance(mesh.elements, CompactSparseTensor):
        f = reshaped_numpy(mesh.elements._indices, [instance, dual])
    else:
        raise NotImplementedError
    print(f"Saving triangle mesh with {v.shape[0]} vertices and {f.shape[0]} faces to {file}")
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez(file, vertices=v, faces=f, f_dim=instance(mesh).name, vertex_dim=instance(mesh.vertices).name, vector=mesh.vector.item_names, has_extra_data=bool(extra_data), **extra_data)


@broadcast
def load_tri_mesh(file: str, convert=False, load_extra=()) -> Union[Mesh, Tuple[Mesh, ...]]:
    data = np.load(file, allow_pickle=bool(load_extra))
    f_dim = instance(str(data['f_dim']))
    vertex_dim = instance(str(data['vertex_dim']))
    vector = channel(vector=[str(d) for d in data['vector']])
    faces = tensor(data['faces'], f_dim, spatial('vertex_list'), convert=convert)
    vertices = tensor(data['vertices'], vertex_dim, vector, convert=convert)
    m = mesh(vertices, faces)
    if not load_extra:
        return m
    extra = [data[e] for e in load_extra]
    extra = [e.tolist() if e.dtype == object else e for e in extra]
    return m, *extra


@broadcast(dims=batch)
def decimate_tri_mesh(mesh: Mesh, factor=.1, target_max=10_000,):
    if isinstance(mesh, NoGeometry):
        return mesh
    if instance(mesh).volume == 0:
        return mesh
    import pyfqmr
    mesh_simplifier = pyfqmr.Simplify()
    vertices = reshaped_numpy(mesh.vertices.center, [instance, 'vector'])
    faces = reshaped_numpy(mesh.elements._indices, [instance, dual])
    target_count = min(target_max, int(round(instance(mesh).volume * factor)))
    mesh_simplifier.setMesh(vertices, faces)
    mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=7, preserve_border=False)
    vertices, faces, normals = mesh_simplifier.getMesh()
    return mesh_from_numpy(vertices, faces, cell_dim=instance(mesh))


@lru_cache
def element_types_3d():
    # Conventions from https://wiki.freecad.org/FEM_Element_Types
    tetra = [(1, 2, 3), (1, 4, 2), (2, 4, 3), (3, 4, 1)]
    pyramid = [(1, 2, 3, 4), (1, 5, 2), (2, 5, 3), (3, 5, 4), (4, 5, 1)]
    prism = [(1, 2, 3), (4, 6, 5), (1, 4, 5, 2), (2, 5, 6, 3), (3, 6, 4, 1, 3)]
    hexa = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    elements = {4: tetra, 5: pyramid, 6: prism, 8: hexa}
    vertices = {k: np.concatenate(v) - 1 for k, v in elements.items()}
    v_count = {k: np.asarray([len(v) for v in e]) for k, e in elements.items()}
    return vertices, v_count
