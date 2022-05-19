import numpy
import numpy as np


def smooth_uniform_curve(curve, n: int):
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


def down_sample_curve(curve: np.ndarray, max_points: int):
    if curve.shape[-2] <= max_points:
        return curve
    step = curve.shape[-2] // max_points
    return np.concatenate([curve[::step], curve[-1:]], -2)
