
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
    return "OK"


def test_torch():
    try:
        import torch
    except ImportError:
        return "Not installed"
    from phi import torch
    with torch.TORCH_BACKEND:
        math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)
    return "OK"


def test_dash():
    try:
        import dash
    except ImportError:
        return "Not installed"
    return 'OK'



math.assert_close(math.ones(batch=8, x=64) + math.ones(batch=8, x=64), 2)

result_tf = test_tensorflow()
result_torch = test_torch()
result_dash = test_dash()


print(f"Installation verified.\nWeb interface: {result_dash}\nTensorFlow: {result_tf}\nPyTorch: {result_torch}")
