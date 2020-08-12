from phi.struct.tensorop import collapsed_gather_nd

from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math


def rank(tensor):
    return len(math.staticshape(tensor))


def spatial_rank(tensor):
    """ The spatial rank of a tensor is ndims - 2. """
    return math.ndims(tensor) - 2


def spatial_dimensions(obj):
    return tuple(range(1, len(math.staticshape(obj)) - 1))


def axes(obj):
    return tuple(range(len(math.staticshape(obj)) - 2))


def all_dimensions(tensor):
    return range(len(math.staticshape(tensor)))


def is_scalar(tensor):
    return math.ndims(tensor) == 0


def _get_pad_width_axes(rank, axes, val_true=(1, 1), val_false=(0, 0)):
    mid_shape = []
    for i in range(rank):
        if _contains_axis(axes, i, rank):
            mid_shape.append(val_true)
        else:
            mid_shape.append(val_false)
    return [[0, 0]] + mid_shape + [[0, 0]]


def _get_pad_width(rank, axis_widths=(1, 1)):
    return [[0, 0]] + [axis_widths] * rank + [[0, 0]]


def _dim_shifted(tensor, axis, relative_shifts, components=None, diminish_others=(0, 0), diminish_other_condition=None):
    assert len(relative_shifts) >= 2
    total_shift = max(relative_shifts) - min(relative_shifts)
    # --- Handle diminish_others ---
    if isinstance(diminish_others, tuple):
        slice_others = slice(diminish_others[0], -diminish_others[1] if diminish_others[1] != 0 else None)
    else:
        raise ValueError("Illegal diminish_others arguemnt: '%s'" % diminish_others)
    # --- Handle components ---
    if components is None:
        component_slice = slice(None)
    elif isinstance(components, int):
        component_slice = slice(components, components + 1)
    elif isinstance(components, slice):
        component_slice = components
    else:
        raise ValueError("Illegal components argument: '%s'" % components)
    # --- Slice tensor to create shifts ---
    rank = spatial_rank(tensor)
    shifted_tensors = []
    for shift in relative_shifts:
        shift_start = shift - min(relative_shifts)
        shift_end = shift_start - total_shift
        if shift_end == 0:
            shift_end = None
        slices = []
        for ax in range(rank):
            if ax == axis:
                slices.append(slice(shift_start, shift_end))
            else:
                if diminish_other_condition is None or diminish_other_condition(ax):
                    slices.append(slice_others)
                else:
                    slices.append(slice(None))
        sliced_tensor = tensor[(slice(None),) + tuple(slices) + (component_slice,)]
        shifted_tensors.append(sliced_tensor)
    return shifted_tensors


def _contains_axis(axes, axis, sp_rank):
    assert -sp_rank <= axis < sp_rank
    return (axes is None) or (axis in axes) or (axis + sp_rank in axes)


def map_for_axes(function, obj, axes, rank):
    if axes is None:
        return function(obj)
    else:
        return [(function(collapsed_gather_nd(obj, i)) if _contains_axis(axes, i, rank)
                 else collapsed_gather_nd(obj, i))
                for i in range(rank)]
