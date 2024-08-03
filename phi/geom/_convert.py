import numpy as np
from phiml import math
from phiml.math import wrap, instance, tensor, dual, batch, DimFilter, unstack
from phiml.math._sparse import CompactSparseTensor
from ._functions import plane_sgn_dist

from ._geom import Geometry
from ._box import Cuboid, BaseBox
from ._sdf import SDF, numpy_sdf
from ._mesh import Mesh, extrinsic_normals
from ._graph import Graph


def as_sdf(geo: Geometry, rel_margin=.1, abs_margin=0., separate: DimFilter = None, method='auto') -> SDF:
    """
    Represent existing geometry as a signed distance function.

    Args:
        geo: `Geometry` to represent as a signed distance function.
            Must implement `Geometry.approximate_signed_distance()`.
        rel_margin: Relative size to pad the domain on all sides around the bounds of `geo`.
            For example, 0.1 will pad 10% of `geo`'s size in each axis on both sides.
        abs_margin: World-space size to pad the domain on all sides around the bounds of `geo`.
        separate: Dimensions along which to unstack `geo` and return individual SDFs.
            Once created, SDFs cannot be unstacked.

    Returns:

    """
    separate = geo.shape.only(separate)
    if separate:
        return math.map(as_sdf, geo, rel_margin, abs_margin, separate=None, dims=separate, unwrap_scalars=True)
    bounds: BaseBox = geo.bounding_box()
    bounds = Cuboid(bounds.center, half_size=bounds.half_size * (1 + 2 * rel_margin) + 2 * abs_margin)
    if isinstance(geo, SDF):
        return SDF(geo._sdf, geo._out_shape, bounds, geo._center, geo._volume, geo._bounding_radius)
    elif isinstance(geo, Mesh) and geo.spatial_rank == 3 and geo.element_rank == 2:  # 3D surface mesh
        method = 'pysdf' if method == 'auto' else method
        if method == 'pysdf':
            from pysdf import SDF as PySDF  # https://github.com/sxyu/sdf    https://github.com/sxyu/sdf/blob/master/src/sdf.cpp
            np_verts = geo.vertices.center.numpy('vertices,vector')
            np_tris = geo.elements._indices.numpy('cells,~vertices')
            np_sdf = PySDF(np_verts, np_tris)  # (num_vertices, 3) and (num_faces, 3)
            np_sdf_c = lambda x: np.clip(np_sdf(x), -float(bounds.size.min) / 2, float(bounds.size.max))
            return numpy_sdf(np_sdf_c, bounds, geo.bounding_box().center)
        elif method == 'closest-face':
            normals = extrinsic_normals(geo)
            face_size = math.sqrt(geo.volume) * 4
            def sdf_closest_face(location):
                closest_elem = math.find_closest(geo.center, location)
                center = geo.center[closest_elem]
                normal = normals[closest_elem]
                return plane_sgn_dist(center, normal, location)
            def sdf_and_grad(location):  # for close distances < face_size use normal vector, for far distances use distance from center
                closest_elem = math.find_closest(geo.center, location)
                center = geo.center[closest_elem]
                normal = normals[closest_elem]
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
    return SDF(geo.approximate_signed_distance, geo.shape.non_instance.without('vector'), bounds, geo.center, geo.volume, geo.bounding_radius())


def surface_mesh(geo: Geometry, rel_dx: float = None, abs_dx: float = None, remove_duplicates=True) -> Mesh:
    """
    Create a surface `Mesh` from a Geometry.

    Args:
        geo: `Geometry` to convert. Must implement `approximate_signed_distance`.
        rel_dx: Relative mesh resolution as fraction of bounding box size.
        abs_dx: Absolute mesh resolution. If both `rel_dx` and `abs_dx` are provided, the lower value is used.
        remove_duplicates: If `False`, mesh may contain duplicate vertices, increasing the stored size.

    Returns:
        `Mesh`
    """
    if geo.spatial_rank != 3:
        raise NotImplementedError("Only 3D SDF currently supported")
    if rel_dx is None and abs_dx is None:
        rel_dx = 0.01
    from sdf.mesh import generate  # https://github.com/fogleman/sdf  pip install git+https://github.com/fogleman/sdf.git
    def generate_mesh(geo: Geometry, rel_dx, abs_dx):
        rel_dx = None if rel_dx is None else rel_dx * geo.bounding_radius().max
        dx = math.minimum(rel_dx, abs_dx, allow_none=True)
        sdf = as_sdf(geo, rel_margin=0, abs_margin=dx)
        lo = sdf.bounds.lower.numpy()
        up = sdf.bounds.upper.numpy()
        def np_sdf(xyz):
            location = wrap(xyz, instance('points'), sdf.shape['vector'])
            sdf_val = sdf._sdf(location)
            return sdf_val.numpy()
        mesh = generate(np_sdf, float(dx), bounds=(lo, up), workers=1, batch_size=1024*1024)
        if not mesh:
            raise ValueError(f"No surface found from SDF in between {lo} and {up}")
        mesh = np.stack(mesh)
        if remove_duplicates:
            vert, idx, inv, c = np.unique(mesh, axis=0, return_counts=True, return_index=True, return_inverse=True)
            vert = tensor(vert, instance('vertex'), sdf.shape['vector'])
            tris_np = inv.reshape((-1, 3))
        else:
            vert = mesh
            raise NotImplementedError  # this is actually the simpler case
            tris_np = np.arange(vert.shape[0]).reshape((-1, 3))
        tris = wrap(tris_np, instance('face'), dual('vertex'))
        tris = CompactSparseTensor(tris, wrap(1), vert.shape['vertex'].as_dual(), True)
        # vert_graph = Graph(vert, None, {})
        return Mesh(vert, tris, 2, {}, None, None, None, None, None, None)
    return math.map(generate_mesh, geo, rel_dx, abs_dx, dims=batch)
