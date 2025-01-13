import numpy as np

from phiml import math
from phiml.math import wrap, instance, batch, DimFilter, Tensor, spatial, pack_dims, dual, stack, to_int32, maximum
from ._box import Cuboid, BaseBox
from ._sphere import Sphere
from ._functions import plane_sgn_dist
from ._geom import Geometry, NoGeometry
from ._mesh import Mesh, mesh_from_numpy, mesh
from ._sdf import SDF, numpy_sdf
from ._sdf_grid import sample_sdf, SDFGrid


def as_sdf(geo: Geometry, bounds=None, rel_margin=None, abs_margin=0., separate: DimFilter = None, method='auto') -> SDF:
    """
    Represent existing geometry as a signed distance function.

    Args:
        geo: `Geometry` to represent as a signed distance function.
            Must implement `Geometry.approximate_signed_distance()`.
        bounds: Bounds of the SDF. If `None` will be determined from bounds of `geo` and `rel_margin`/`abs_margin`.
        rel_margin: Relative size to pad the domain on all sides around the bounds of `geo`.
            For example, 0.1 will pad 10% of `geo`'s size in each axis on both sides.
        abs_margin: World-space size to pad the domain on all sides around the bounds of `geo`.
        separate: Dimensions along which to unstack `geo` and return individual SDFs.
            Once created, SDFs cannot be unstacked.

    Returns:

    """
    separate = geo.shape.only(separate)
    if separate:
        return math.map(as_sdf, geo, bounds, rel_margin, abs_margin, separate=None, dims=separate, unwrap_scalars=True)
    if bounds is None:
        bounds: BaseBox = geo.bounding_box()
        rel_margin = .1 if rel_margin is None else rel_margin
    rel_margin = 0 if rel_margin is None else rel_margin
    bounds = Cuboid(bounds.center, half_size=bounds.half_size * (1 + 2 * rel_margin) + 2 * abs_margin)
    if isinstance(geo, SDF):
        return SDF(geo._sdf, geo._out_shape, bounds, geo._center, geo._volume, geo._bounding_radius)
    elif isinstance(geo, Mesh) and geo.spatial_rank == 3 and geo.element_rank == 2:  # 3D surface mesh
        method = 'closest-face' if method == 'auto' else method
        if method == 'pysdf':
            from pysdf import SDF as PySDF  # https://github.com/sxyu/sdf    https://github.com/sxyu/sdf/blob/master/src/sdf.cpp
            np_verts = geo.vertices.center.numpy('vertices,vector')
            np_tris = geo.elements._indices.numpy('cells,~vertices')
            np_sdf = PySDF(np_verts, np_tris)  # (num_vertices, 3) and (num_faces, 3)
            np_sdf_c = lambda x: np.clip(np_sdf(x), -float(bounds.size.min) / 2, float(bounds.size.max))
            return numpy_sdf(np_sdf_c, bounds, geo.bounding_box().center)
        elif method == 'closest-face':
            def sdf_closest_face(location):
                closest_elem = math.find_closest(geo.center, location)
                center = geo.center[closest_elem]
                normal = geo.normals[closest_elem]
                return plane_sgn_dist(center, normal, location)
            def sdf_and_grad(location):  # for close distances < face_size use normal vector, for far distances use distance from center
                closest_elem = math.find_closest(geo.center, location)
                center = geo.center[closest_elem]
                normal = geo.normals[closest_elem]
                face_size = math.sqrt(geo.volume) * 4
                size = face_size[closest_elem]
                sgn_dist = plane_sgn_dist(center, normal, location)
                outward = math.where(abs(sgn_dist) < size, normal, math.vec_normalize(location - center))
                return sgn_dist, outward
            return SDF(sdf_closest_face, math.EMPTY_SHAPE, bounds, geo.bounding_box().center, sdf_and_grad=sdf_and_grad)
        elif method == 'mesh-to-sdf':
            from mesh_to_sdf import mesh_to_sdf
            from trimesh import Trimesh
            np_verts = geo.vertices.center.numpy('vertices,vector')
            np_tris = geo.elements._indices.numpy('cells,~vertices')
            trimesh = Trimesh(np_verts, np_tris)
            def np_sdf(points):
                return mesh_to_sdf(trimesh, points, surface_point_method='scan', sign_method='normal')
            return numpy_sdf(np_sdf, bounds, geo.bounding_box().center)
        else:
            raise ValueError(f"Method '{method}' not implemented for Mesh SDF")
    def sdf_and_grad(x: Tensor):
        sgn_dist, delta, *_ = geo.approximate_closest_surface(x)
        return sgn_dist, math.vec_normalize(-delta)
    return SDF(geo.approximate_signed_distance, geo.shape.non_instance.without('vector'), bounds, geo.center, geo.volume, geo.bounding_radius(), sdf_and_grad)


def surface_mesh(geo: Geometry,
                 rel_dx: float = None,
                 abs_dx: float = None,
                 method='auto') -> Mesh:
    """
    Create a surface `Mesh` from a Geometry.

    Args:
        geo: `Geometry` to convert. Must implement `approximate_signed_distance`.
        rel_dx: Relative mesh resolution as fraction of bounding box size.
        abs_dx: Absolute mesh resolution. If both `rel_dx` and `abs_dx` are provided, the lower value is used.
        method: 'auto' to select based on the type of `geo`. 'lewiner' or 'lorensen' for marching cubes.

    Returns:
        `Mesh` if there is any geometry
    """
    if geo.spatial_rank != 3:
        raise NotImplementedError("Only 3D SDF currently supported")
    if isinstance(geo, NoGeometry):
        return mesh_from_numpy([], [], element_rank=2)
    # --- Determine resolution ---
    if isinstance(geo, SDFGrid):
        assert rel_dx is None and abs_dx is None, f"When creating a surface mesh from an SDF grid, rel_dx and abs_dx are determined from the grid and must be specified as None"
    if rel_dx is None and abs_dx is None:
        rel_dx = 1 / 128
    rel_dx = None if rel_dx is None else rel_dx * geo.bounding_box().size.max
    dx = math.minimum(rel_dx, abs_dx, allow_none=True)
    # --- Check special cases ---
    if method == 'auto' and isinstance(geo, BaseBox):
        assert rel_dx is None and abs_dx is None, f"When method='auto', boxes will always use their corners as vertices. Leave rel_dx,abs_dx unspecified or pass 'lewiner' or 'lorensen' as method"
        vertices = pack_dims(geo.corners, dual, instance('vertices'))
        corner_count = vertices.vertices.size
        vertices = pack_dims(vertices, instance(geo) + instance('vertices'), instance('vertices'))
        v1 = [0, 1, 4, 5, 4, 6, 5, 7, 0, 1, 2, 6]
        v2 = [1, 3, 6, 6, 0, 0, 7, 3, 4, 4, 3, 3]
        v3 = [2, 2, 5, 7, 6, 2, 1, 1, 1, 5, 6, 7]
        instance_offset = math.range_tensor(instance(geo)) * corner_count
        faces = wrap([v1, v2, v3], spatial('vertices'), instance('faces')) + instance_offset
        faces = pack_dims(faces, instance, instance('faces'))
        return mesh(vertices, faces, element_rank=2)
    elif method == 'auto' and isinstance(geo, Sphere):
        pass  # ToDo analytic solution
    # --- Build mesh from SDF ---
    if isinstance(geo, SDFGrid):
        sdf_grid = geo
    else:
        if isinstance(geo, SDF):
            sdf = geo
        else:
            sdf = as_sdf(geo, rel_margin=0, abs_margin=dx)
        resolution = maximum(1, to_int32(math.round(sdf.bounds.size / dx)))
        resolution = spatial(**resolution.vector)
        sdf_grid = sample_sdf(sdf, sdf.bounds, resolution)
    from skimage.measure import marching_cubes
    method = 'lewiner' if method == 'auto' else method
    def generate_mesh(sdf_grid: SDFGrid) -> Mesh:
        dx = sdf_grid.dx.numpy()
        sdf_numpy = sdf_grid.values.numpy(sdf_grid.dx.vector.item_names)
        vertices, faces, v_normals, _ = marching_cubes(sdf_numpy, level=0.0, spacing=dx, allow_degenerate=False, method=method)
        vertices += sdf_grid.bounds.lower.numpy() + .5 * dx
        with math.NUMPY:
            return mesh_from_numpy(vertices, faces, element_rank=2, cell_dim=instance('faces'))
    return math.map(generate_mesh, sdf_grid, dims=batch)
