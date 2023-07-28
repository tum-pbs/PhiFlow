import numpy
import numpy as np

from phiml.math import Tensor, spatial, ones, convolve, extrapolation, map_c2b


def smooth_uniform_curve(curve, n: int):
    # deprecated, use smooth() instead
    if n == 1:
        return curve
    x = curve[..., 0]
    values = curve[..., 1]
    if n == -1 or curve.shape[0] <= n:  # mean value
        mean = numpy.mean(values, -1, keepdims=True)
        const_curve = numpy.array([(numpy.min(x), mean), (numpy.max(x), mean)], dtype=np.float32)
        return const_curve
    else:  # smooth kernel
        result = np.convolve(values, np.ones((n,)) / n, mode='valid')
        valid = x[n//2-1:-n//2 or None]
        return np.stack([valid, result], -1)


@map_c2b
def smooth(curves: Tensor, n: int) -> Tensor:
    """
    Applies a smoothing kernel to curves, all channels independently.

    Args:
        curves: `Tensor` containing at least one spatial dimension
        n: Kernel size, i.e. number of values to average.

    Returns:
        Smoothed curves as `Tensor`
    """
    assert isinstance(n, int), f"n must be an int but got {n}"
    assert n >= 1, f"n must be at least 1 but got {n}"
    if n == 1:
        return curves
    kernel = ones(spatial(curves).with_sizes(n)) / n ** spatial(curves).rank
    return convolve(curves, kernel, extrapolation=extrapolation.SYMMETRIC_GRADIENT)



def down_sample_curve(curve: np.ndarray, max_points: int):
    if curve.shape[-2] <= max_points:
        return curve
    step = curve.shape[-2] // max_points
    return np.concatenate([curve[::step], curve[-1:]], -2)
