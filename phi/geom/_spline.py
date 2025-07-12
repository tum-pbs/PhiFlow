from typing import Sequence, Union, Dict, List

import numpy as np

from phiml import math, Shape, wrap, minimum, maximum, mean, batch
from phiml.math import Tensor, spatial, dual, stack, clip, channel, to_int32, meshgrid, unstack, cpack
from ._functions import cross, vec_normalize, solve2x2


def b_spline_knots(bases_dim: Shape, degree: int, curve_type='clamped', crease: Union[Sequence[float], Tensor] = None, pad_to: int = None) -> Tensor:
    """
    Compute knot matrix for B-spline.

    Args:
        bases_dim: Dimension listing the basis functions, size equal to number of control points.
        degree: B-spline degree, 1 for linear, 2 for quadratic, 3 for cubic.
        curve_type: 'clamped' for curve passing through endpoints, 'uniform' for uniform knots.
        crease: Values in the range [0,1] for each inner control point. Larger values add more weight / multiplicity to a control point. A value of 1 makes the spline pass through that control point, but the spline loses differentiability at that point.
        pad_to: Pad knot tensor to `pad_to` basis functions. This enables uniform stacking of knot tensors of splines with different resolutions.
    """
    n = bases_dim.size
    if curve_type == 'clamped':
        # Clamped knot vector - curve passes through first and last control points
        knots = np.zeros(n + degree + 1)
        knots[:degree + 1] = 0.0
        knots[-(degree + 1):] = 1.0
        # Internal knots
        if n > degree + 1:
            internal_knots = np.linspace(0, 1, n - degree + 1)[1:-1]
            knots[degree + 1:n] = internal_knots
    elif curve_type == 'uniform':
        # Uniform knot vector
        knots = np.linspace(0, 1, n + degree + 1)
    knot_slices = [knots[i:i+n] for i in range(degree+2)]
    knot_matrix = wrap(np.asarray(knot_slices), 'support:s', bases_dim)
    # --- Apply crease ---
    if crease is not None:
        centers = mean(knot_matrix[{bases_dim: slice(1, -1)}].support[1:-1], 'support')  # approximate mode of each basis function
        crease_idx = range(1, len(crease) + 1)
        basis_idx = math.range(bases_dim)
        for c, i, cen in zip(crease, crease_idx, centers):  # for each crease position, compute knot shift of all basis functions
            is_basis_left = basis_idx < i
            is_basis_right = basis_idx > i
            crease1_knots = math.where(is_basis_left, minimum(knot_matrix, cen), knot_matrix)
            crease1_knots = math.where(is_basis_right, maximum(crease1_knots, cen), crease1_knots)
            knot_matrix = c * crease1_knots + (1-c) * knot_matrix
    if pad_to is not None and pad_to > n:
        knot_matrix = math.pad(knot_matrix, {bases_dim.name: (0, pad_to - n)}, 1)
    return knot_matrix


def eval_nurbs_bases(t: Tensor, knots: Tensor, weights: Tensor = None, compute_derivative=False, eps=1e-5) -> Tensor:
    """
    Compute all NURBS basis functions.
    This simplifies to B-spline basis functions if `weights=None` and knots are uniform.

    Args:
        t: Parameter value where to evaluate the basis functions.
        knots: Knot matrix of shape (~bases:d, support:s=degree+2).
        weights: NURBS weight per control point. Shape (~bases:d,)
        eps: Value smaller than 1/n, ensuring that the upper end t=1.0 is handled correctly.

    Returns:
        Basis function values at `t` of all basis function listed along `bases_dim`.
    """
    bases_dim = spatial(knots) & dual(knots)
    degree = knots.shape.get_size('support') - 2
    supports = tuple(knots.support)
    t_c = math.clip(t, 0, 1 - eps)  # Clamped version of t with t=1.0 belonging to last knot
    bases = [tuple(math.to_float((t_c >= knots.support[:-1]) & (t_c < knots.support[1:])).support)]
    for deg in range(1, degree+1):
        bases_i = []
        for offset in range(degree - deg + 1):
            denom1 = supports[offset + deg] - supports[offset]
            term1 = bases[-1][offset] * math.safe_div(t - supports[offset], denom1)
            knots_deg1 = supports[offset + deg + 1]
            denom2 = knots_deg1 - supports[offset + 1]
            term2 = bases[-1][offset + 1] * math.safe_div(knots_deg1 - t, denom2)
            bases_i.append(term1 + term2)
        bases.append(bases_i)
    val = bases[-1][0] if weights is None else bases[-1][0] * weights
    norm = math.sum(val, bases_dim)
    if compute_derivative:
        dw_dt = degree * (math.safe_div(bases[-2][0], denom1) - math.safe_div(bases[-2][1], denom2))
        norm_dt = math.sum(dw_dt, bases_dim)
        derivative = dw_dt / norm - val / norm**2 * norm_dt  # correction for shape change due to normalization
        return stack([val / norm, derivative], batch('d_order'))
    return val / norm
