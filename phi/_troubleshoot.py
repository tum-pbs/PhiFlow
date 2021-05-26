from contextlib import contextmanager


def assert_minimal_config():  # raises AssertionError
    import sys
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "phiflow requires Python 3.6 or newer to run"

    try:
        import numpy
    except ImportError:
        raise AssertionError("phiflow is unable to run because NumPy is not installed.")
    try:
        import scipy
    except ImportError:
        raise AssertionError("phiflow is unable to run because SciPy is not installed.")
    from phi import flow
    from phi import math
    with math.NUMPY_BACKEND:
        a = math.ones(batch=8, x=64)
        math.assert_close(a + a, 2)


def troubleshoot():
    import phi
    return f"PhiFlow version {phi.__version__}\n"\
           f"Web interface: {troubleshoot_dash()}\n"\
           f"TensorFlow: {troubleshoot_tensorflow()}\n"\
           f"PyTorch: {troubleshoot_torch()}\n"\
           f"Jax: {troubleshoot_jax()}"


def troubleshoot_tensorflow():
    from phi import math
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors
    try:
        import tensorflow
    except ImportError:
        return "Not installed."
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
            # TODO cuDNN math.convolve(math.ones(batch=8, x=64), math.ones(x=4))
        except BaseException as err:
            return f"Installed but tests failed with error: {err}"
    if gpu_count == 0:
        return f"Installed, {gpu_count} GPUs available."
    else:
        from phi.tf._tf_cuda_resample import librariesLoaded
        if librariesLoaded:
            cuda_str = 'CUDA kernels available.'
        else:
            import platform
            if platform.system().lower() != 'linux':
                cuda_str = f"CUDA kernels not available and compilation not recommended on {platform.system()}."
            else:
                cuda_str = f"CUDA kernels not available. Clone the phiflow source from GitHub and run 'python setup.py tf_cuda' to compile them."
        return f"Installed, {gpu_count} GPUs available. {cuda_str}"


def troubleshoot_torch():
    from phi import math
    try:
        import torch
    except ImportError:
        return "Not installed."
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


def troubleshoot_jax():
    from phi import math
    try:
        import jax
        import jaxlib
    except ImportError:
        return "Not installed."
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


def troubleshoot_dash():
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
        return "ImageIO not installed, 3D view might not work."
    try:
        import matplotlib
    except ImportError:
        return "Matplotlib not installed"
    try:
        dash.Dash('Test')
    except BaseException as e:
        return f"Runtime error: {e}"
    return 'OK'


@contextmanager
def plot_solves():
    """


    While `plot_solves()` is active, certain performance optimizations and algorithm implementations may be disabled.

    Returns:

    """
    from .math import SolveTape, l1_loss, SolveInfo
    import pylab
    with SolveTape(record_trajectories=True) as solves:
        try:
            yield solves
        finally:
            for i, result in enumerate(solves):
                assert isinstance(result, SolveInfo)
                res = l1_loss(result.residual)
                pylab.plot(res.inflow_loc[0].numpy(), label=f"{i}: {result.method}")
                print(f"Solve {i}: {result.method} ({1000 * result.solve_time:.1f} ms)\n"
                      f"\t{result.solve}\n"
                      f"\t{result.msg}\n"
                      f"\tConverged: {result.converged}\n"
                      f"\tDiverged: {result.diverged}\n"
                      f"\tIterations: {result.iterations}\n"
                      f"\tFunction evaulations: {result.function_evaluations.trajectory[-1]}")
            pylab.yscale('log')
            pylab.ylabel("Residual $L_1$")
            pylab.xlabel("Iteration")
            pylab.title(f"Solve Convergence")
            pylab.legend(loc='upper right')
            pylab.savefig(f"pressure-solvers-FP32.png")
            pylab.show()
