from phi.math import EMPTY_SHAPE
from ._magic_ops import concat
from ._ops import mean, ones, reshaped_tensor, reshaped_native
from ._shape import DimFilter, instance, shape, channel
from ._tensors import Tensor


def fit_line_2d(x: Tensor, y: Tensor, point_dim: DimFilter = instance, weights: Tensor = 1.):
    """
    Fits a line of the form *slope · x + offset* to pass through the data points defined by their coordinates `x` and `y`.

    Args:
        x: X coordinate of the points.
        y: Y coordinate of the points.
        point_dim: Dimension listing the points the line should pass through. This dimension will be reduced in the operation.
            By default, all instance dimensions
        weights: (Optional) Tensor assigning a weight to each point in `x` and `y` to determine the relative influence of that point in the overall fit.

    Returns:
        slope: Line slope in units y/x as `Tensor`
        offset: Line value for x=0.
    """
    assert shape(x).only(point_dim) or shape(y).only(point_dim), f"Either x or y need to have a dimension corresponding to point_dim but got x: {shape(x)}, y: {shape(y)}"
    if not shape(weights):  # unweighted fit
        mean_x = mean(x, point_dim)
        x_rel = x - mean_x
        var_x = mean(x_rel ** 2, point_dim)
        slope = mean(x_rel * y, point_dim) / var_x
        offset = mean(y, point_dim) - slope * mean_x
    else:  # weighted fit
        mean_w = mean(weights, point_dim)
        mean_x = mean(weights * x, point_dim) / mean_w
        x_rel = x - mean_x
        var_wx = mean(weights * x_rel ** 2, point_dim)
        slope = mean(weights * x_rel * y, point_dim) / var_wx
        offset = mean(weights * y, point_dim) / mean_w - slope * mean_x
    return slope, offset


def fit_hyperplane(x: Tensor, y: Tensor, point_dim: DimFilter = instance, weights: Tensor = 1.):
    """
    Fits an n-dimensional plane through the points (*x, y).

    Args:
        x: `Tensor` containing `point_dim` and a channel dimensions for the vector components.
        y: `Tensor` containing `point_dim`
        point_dim: Dimension listing the points the hyperplane should pass through. This dimension will be reduced in the operation.
            By default, all instance dimensions
        weights: (Optional) Tensor assigning a weight to each point in `x` and `y` to determine the relative influence of that point in the overall fit.

    Returns:
        slope: Plane slope in units y/x as `Tensor`
        offset: Plane value for x=0.
    """
    point_dim = shape(x).only(point_dim)
    assert point_dim.rank == 1
    vec_dim = channel(x).without(point_dim)
    assert vec_dim.rank == 1, f"x must have a channel dimension for to encode vectors but has shape {shape(x)}"
    assert vec_dim not in shape(y)
    batch_dims = (shape(x).without(vec_dim) & shape(y)).without(point_dim)
    assert point_dim not in shape(weights), f"Weights may not contain the vector/features dimension {point_dim}."
    assert vec_dim.rank == 1, f"x must have exactly 1 channel dimension (excluding point_dim) to act as the vector dimension listing the components/features but got shape {shape(x)}"
    mat = concat([x, ones(shape(x).without(vec_dim), channel(**{vec_dim.name: 'y'}))], vec_dim) * weights
    y *= weights
    # Least Squares fit
    np_mat = reshaped_native(mat, [batch_dims, point_dim, vec_dim.name], force_expand=True)
    np_rhs = reshaped_native(y, [batch_dims, point_dim, '_batch_per_matrix'], force_expand=True)
    from phi.math.backend import choose_backend
    backend = choose_backend(np_mat, np_rhs)
    solution, *_ = backend.matrix_solve_least_squares(np_mat, np_rhs)
    slope = reshaped_tensor(solution[..., :-1, :], [batch_dims, vec_dim, EMPTY_SHAPE])
    offset = reshaped_tensor(solution[..., -1, :], [batch_dims, EMPTY_SHAPE])
    return slope, offset
