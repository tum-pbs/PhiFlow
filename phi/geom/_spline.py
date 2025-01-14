from typing import Sequence, Union, Dict

from phiml import math
from phiml.math import Tensor, spatial, dual, stack, clip, channel, to_int32, meshgrid, unstack, cpack
from ._functions import cross, vec_normalize, solve2x2


def spline_eval(order, points, uv: Tensor, get: Sequence[str] = ('position', 'tangents', 'normal')) -> Sequence[Tensor]:
    idx0 = clip(to_int32(uv), 0, spatial(points) - 2)
    ruv = uv - idx0
    indices = idx0 + meshgrid(spatial(points).with_sizes(2))
    nb_points = points[indices]
    u, v = spatial(points).names
    w_u = stack([1-ruv[u], ruv[u]], dual(u))
    w_v = stack([1-ruv[v], ruv[v]], dual(v))
    weights = w_u * w_v
    result = {}
    if 'position' in get:
        result['position'] = weights @ nb_points
    if 'tangents' in get or 'normal' in get:
        tangents = {}
        for dim in spatial(points):
            other_dim = spatial(points) - dim
            ruv_dim = ruv[other_dim.name]
            lo_tan, up_tan = unstack(nb_points[{dim: 1}] - nb_points[{dim: 0}], spatial(points)-dim)
            tangent = ruv_dim * up_tan + (1-ruv_dim) * lo_tan
            tangents[dim] = tangent
        if 'tangents' in get:
            result['tangents'] = stack(tangents, '~tangents')
        if 'normal' in get:
            result['normal'] = vec_normalize(cross(*tangents.values()))
    return [result[t] for t in get]  # re-order output to match input+


def closest_param(order: Union[int, Dict[str, int]], points, location: Tensor, dist_iter=3):
    assert order == 1 or isinstance(order, dict) and all(v == 1 for v in order.values())
    assert dist_iter > 0
    idx = to_int32(math.find_closest(points, location, index_dim=channel('spline')))
    uv = idx.spline.rename('vector')
    for i in range(dist_iter):
        center, tangents = spline_eval(1, points, uv, ('position', 'tangents'))
        proj = tangents.vector @ (location - center)
        tu, tv = tangents.tangents.dual
        tan_dot = tu.vector @ tv.vector
        uu, vv = (tangents.vector @ tangents.vector).tangents.dual
        uv += solve2x2(uu, tan_dot, tan_dot, vv, *proj.tangents.dual)
    buv = clip(uv, 0, spatial(points) - 1)
    on_surface, = spline_eval(1, points, buv, ('position',))
    buv = cpack(buv, channel(spline=spatial(points).name_list))
    fuv = cpack(uv, channel(spline=spatial(points).name_list))
    return on_surface, buv, fuv, tangents
