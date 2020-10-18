from ._tensors import Tensor
from .backend.tensorop import collapsed_gather_nd


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


def _multi_roll(tensor, roll_name, relative_shifts, diminish_others=(0, 0), names=None, base_selection={}):
    assert isinstance(tensor, Tensor), tensor
    assert len(relative_shifts) >= 2
    total_shift = max(relative_shifts) - min(relative_shifts)
    slice_others = slice(diminish_others[0], -diminish_others[1] if diminish_others[1] != 0 else None)
    # --- Slice tensor to create shifts ---
    shifted_tensors = []
    for shift in relative_shifts:
        slices = dict(base_selection)
        for name in names:
            if name == roll_name:
                shift_start = shift - min(relative_shifts)
                shift_end = shift_start - total_shift
                if shift_end == 0:
                    shift_end = None
                slices[name] = slice(shift_start, shift_end)
            else:
                slices[name] = slice_others
        sliced_tensor = tensor[slices]
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
