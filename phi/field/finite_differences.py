from phiml import spatial, stack
from phiml.math import Extrapolation, spatial_gradient, neighbor_mean, DimFilter
from ._field import Field


def central_gradient_from_nodes(node_field: Field, dims=spatial, boundary: Extrapolation = None, stack_dim='~vector', reduce: DimFilter = None):
    assert node_field.sampled_at == 'node', f"Input Field must be sampled at nodes but is actually sampled at {node_field.sampled_at}"
    dims = spatial(node_field).only(dims, reorder=True)
    reduce = node_field.values.shape.only(reduce)
    grads = {}
    for dim in dims.names:
        values = node_field.values
        if reduce:
            values = values[{reduce.name: dim}]
        grad = spatial_gradient(values, dims=dim, dx=node_field.dx[dim], difference='forward', padding=None, stack_dim=None)
        grads[dim] = neighbor_mean(grad, dims - dim, padding=None)
    grad_values = stack(grads, stack_dim)
    if boundary is not None:
        boundary = node_field.boundary.spatial_gradient()
    return Field(node_field.geometry, grad_values, boundary)


def nodel_gradient_from_centroids(centroid_field: Field, dims=spatial, boundary: Extrapolation = None, stack_dim='~vector', reduce: DimFilter = None) -> Field:
    assert centroid_field.sampled_at == 'center', f"Input Field must be sampled at centers but is actually sampled at {centroid_field.sampled_at}"
    dims = spatial(centroid_field).only(dims, reorder=True)
    reduce = centroid_field.values.shape.only(reduce)
    result = {}
    for dim in dims.names:
        values = centroid_field.values
        if reduce:
            values = values[{reduce.name: dim}]
        grad_staggered = spatial_gradient(values, centroid_field.dx[dim], 'forward', 0, dim, pad=(1, 0), stack_dim=None)
        grad_nodes = neighbor_mean(grad_staggered, dims - dim, padding=0, extend_bounds=(1, 0))
        result[dim] = grad_nodes
    result = stack(result, stack_dim)
    if boundary is not None:
        boundary = centroid_field.boundary.spatial_gradient()
    return Field(centroid_field.geometry, result, boundary)
