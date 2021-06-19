import numpy


def smooth_uniform_curve(x, values, n=16):
    if n == 1:
        return x, values
    if len(x) <= n:
        mean = numpy.tile(numpy.mean(values, -1, keepdims=True), 2)
        return numpy.array([numpy.min(x), numpy.max(x)]), mean
    arrays = [values[i:i-n+1 or None] for i in range(n)]
    result = numpy.mean(arrays, axis=0)
    return x[n//2-1:-n//2 or None], result
