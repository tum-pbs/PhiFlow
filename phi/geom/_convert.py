import numpy as np
from phiml import math
from phiml.math import wrap, instance, tensor, dual, batch
from phiml.math._sparse import CompactSparseTensor

from ._geom import Geometry
from ._box import Cuboid, BaseBox
from ._sdf import SDF
from ._mesh import Mesh
from ._graph import Graph


def as_sdf(geo: Geometry, rel_margin=.1, abs_margin=0.) -> SDF:
    bounds: BaseBox = geo.bounding_box()
    bounds = Cuboid(bounds.center, half_size=bounds.half_size * (1 + 2 * rel_margin) + 2 * abs_margin)
    if isinstance(geo, SDF):
        return SDF(geo._sdf, geo._out_shape, bounds, geo._center, geo._volume, geo._bounding_radius)
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
        mesh = np.stack(generate(np_sdf, float(dx), bounds=(lo, up), workers=1, batch_size=1024*1024))
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
