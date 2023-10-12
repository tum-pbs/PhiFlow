"""
Alias for [`phiml.math`](https://tum-pbs.github.io/PhiML/phiml/math/index.html).

Φ-Flow builds on the tensor functionality from [Φ-ML](https://github.com/tum-pbs/PhiML).
Its `math` package is re-imported here for convenience.
"""
from phiml.math import *

divide_no_nan = safe_div
isfinite = is_finite
vec_abs = vec_length
functional_gradient = gradient
