from ._shape import DimFilter, instance, shape
from ._tensors import Tensor
from ._ops import mean


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
