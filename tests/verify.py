import sys
if sys.version_info.major < 3 \
        or sys.version_info.major == 3 and sys.version_info.minor < 7:
    print("phiflow requires Python 3.7 or newer to run", file=sys.stderr)
    exit(1)

try:
    import numpy
except ImportError:
    print('phiflow is unable to run because NumPy is not installed.', file=sys.stderr)
    exit(1)
try:
    import scipy
except ImportError:
    print('phiflow is unable to run because SciPy is not installed.', file=sys.stderr)
    exit(1)

import phi
from phi.flow import *


def test_tensorflow():
    try:
        import tensorflow
    except ImportError:
        return "Not installed"
    try:
        from phi import tf
    except BaseException as err:
        return f"Installed but not available due to internal error: {err}"
    try:
        gpu_count = len(tf.TF_BACKEND.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with tf.TF_BACKEND:
        try:
            math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
            # TODO cuDNN math.conv(math.ones(batch=8, x=64), math.ones(x=4))
        except BaseException as err:
            return f"Installed but tests failed with error: {err}"
    return f"Installed, {gpu_count} GPUs available."


def test_torch():
    try:
        import torch
    except ImportError:
        return "Not installed"
    try:
        from phi import torch
    except BaseException as err:
        return f"Installed but not available due to internal error: {err}"
    try:
        gpu_count = len(torch.TORCH_BACKEND.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with torch.TORCH_BACKEND:
        try:
            math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
        except BaseException as err:
            return f"Installed but tests failed with error: {err}"
    return f"Installed, {gpu_count} GPUs available."


def test_jax():
    try:
        import jax
        import jaxlib
    except ImportError:
        return "Not installed"
    try:
        from phi import jax
    except BaseException as err:
        return f"Installed but not available due to internal error: {err}"
    try:
        gpu_count = len(jax.JAX_BACKEND.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with jax.JAX_BACKEND:
        try:
            math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
        except BaseException as err:
            return f"Installed but tests failed with error: {err}"
    return f"Installed, {gpu_count} GPUs available."


def test_dash():
    try:
        import dash
    except ImportError:
        return "Dash not installed"
    try:
        import plotly
    except ImportError:
        return "Plotly not installed"
    try:
        import imageio
    except ImportError:
        return "ImageIO not installed"
    try:
        import matplotlib
    except ImportError:
        return "Matplotlib not installed"
    try:
        dash.Dash('Test')
    except BaseException as e:
        return f"Runtime error: {e}"
    return 'OK'


math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)

result_dash = test_dash()
result_tf = test_tensorflow()
result_torch = test_torch()
result_jax = test_jax()


print(f"\nInstallation verified. PhiFlow version {phi.__version__}\n"
      f"Web interface: {result_dash}\n"
      f"TensorFlow: {result_tf}\n"
      f"PyTorch: {result_torch}\n"
      f"Jax: {result_jax}")
