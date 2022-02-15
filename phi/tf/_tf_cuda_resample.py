import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError

# Load Custom Ops
librariesLoaded = False
try:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    resample_op_path = os.path.join(current_dir, "cuda/build/resample.so")
    resample_gradient_op_path = os.path.join(
        current_dir, "cuda/build/resample_gradient.so"
    )
    assert os.path.isfile(resample_op_path), (
        'CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them'
        % resample_op_path
    )
    assert os.path.isfile(resample_gradient_op_path), (
        'CUDA binaries not found at %s. Run "python setup.py tf_cuda" to '
        "compile them" % resample_gradient_op_path
    )
    resample_op = tf.load_op_library(resample_op_path)
    resample_gradient_op = tf.load_op_library(resample_gradient_op_path)
    librariesLoaded = True
except (RuntimeError, AssertionError) as e:
    librariesLoaded = False
except NotFoundError as e:
    # e.g.: tensorflow.python.framework.errors_impl.NotFoundError: libcudart.so.10.0: cannot open shared object file: No such file or directory
    librariesLoaded = False
    print(f"Could not find resample library.  {e}")
except Exception as e:
    print(e)
    librariesLoaded = False

# Register spatial_gradient


@ops.RegisterGradient("Resample")
def _resample_gradient(op, gradient):
    gradients = resample_gradient_op.resample_gradient(
        gradient, op.inputs[0], op.inputs[1], op.inputs[2]
    )
    return [gradients[0], gradients[1], None]


def use_cuda(inputs):
    if not librariesLoaded:
        return False
    if not tf.test.is_gpu_available(True, (3, 0)):
        return False
    shape = inputs.shape
    dims = len(shape) - 2
    components = shape[len(shape) - 1]
    if dims > 3 or components > 4:
        return False
    if dims == 1 and shape[1] > 8192:
        return False
    if dims == 2 and (shape[1] > 32768 or shape[2] > 65536):
        return False
    if dims == 3 and (shape[1] > 2048 or shape[2] > 2048 or shape[3] > 2048):
        return False
    return True


EXT_ID = {'undefined': 0, 'zeros': 0, 'boundary': 1, 'periodic': 2, 'symmetric': 3, 'reflect': 4}


def resample_cuda(inputs, sample_coords, extrapolation):
    shape = inputs.shape
    dims = len(shape) - 2
    boundary_array = np.zeros((dims, 2), np.uint32)
    for i in range(dims):
        for j in range(2):
            current_extrapolation = collapsed_gather_nd(extrapolation, [i, j]).lower()
            if current_extrapolation in EXT_ID:
                boundary_array[i, j] = EXT_ID[current_extrapolation]
            else:
                return NotImplemented
    return resample_op.resample(inputs, sample_coords, boundary_array)


def collapsed_gather_nd(collapsed, nd_index, leaf_condition=None):
    if isinstance(collapsed, (tuple, list, np.ndarray)):
        if leaf_condition is not None and leaf_condition(collapsed):
            return collapsed
        # collapsed = np.array(collapsed)
        if len(nd_index) == 1:
            return collapsed[nd_index[0]]
        else:
            return collapsed_gather_nd(collapsed[nd_index[0]], nd_index[1:])
    else:
        return collapsed
