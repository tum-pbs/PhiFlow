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
        import torch
    except ImportError:
        return "Not installed"
    from phi import tf
    with tf.TF_BACKEND:
        math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
    # TODO cuDNN math.conv(math.ones(batch=8, x=64), math.ones(x=4))
    gpu_count = len(tf.TF_BACKEND.list_devices('GPU'))
    return f"Installed, {gpu_count} GPUs available."


def test_torch():
    try:
        import torch
    except ImportError:
        return "Not installed"
    from phi import torch
    with torch.TORCH_BACKEND:
        math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
    gpu_count = len(torch.TORCH_BACKEND.list_devices('GPU'))
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

result_tf = test_tensorflow()
result_torch = test_torch()
result_dash = test_dash()


print(f"Installation verified. phiFlow version {phi.__version__}\nWeb interface: {result_dash}\nTensorFlow: {result_tf}\nPyTorch: {result_torch}")
