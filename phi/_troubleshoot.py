from contextlib import contextmanager

from phi.math import batch, spatial


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
    with math.NUMPY:
        a = math.ones(batch(batch=8) & spatial(x=64))
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
        import tensorflow_probability
    except ImportError:
        return "Module 'tensorflow_probability' missing. Some functions may be unavailable, such as math.median() and math.quantile(). To install it, run  $ pip install tensorflow-probability"
    try:
        from phi import tf
    except BaseException as err:
        return f"Installed but not available due to internal error: {err}"
    try:
        gpu_count = len(tf.TENSORFLOW.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with tf.TENSORFLOW:
        try:
            math.assert_close(math.ones(batch(batch=8) & spatial(x=64)) + math.ones(batch(batch=8) & spatial(x=64)), 2)
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
        gpu_count = len(torch.TORCH.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with torch.TORCH:
        try:
            math.assert_close(math.ones(batch(batch=8) & spatial(x=64)) + math.ones(batch(batch=8) & spatial(x=64)), 2)
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
        gpu_count = len(jax.JAX.list_devices('GPU'))
    except BaseException as err:
        return f"Installed but device initialization failed with error: {err}"
    with jax.JAX:
        try:
            math.assert_close(math.ones(batch(batch=8) & spatial(x=64)) + math.ones(batch(batch=8) & spatial(x=64)), 2)
        except BaseException as err:
            return f"Installed but tests failed with error: {err}"
    return f"Installed, {gpu_count} GPUs available."


def troubleshoot_dash():
    try:
        import dash
    except ImportError:
        return "Dash not installed. Will fallback to console interface. To install dash, run  $ pip install dash"
    try:
        import plotly
    except ImportError:
        return "Plotly not installed. This package is required by dash. To install it, run  $ pip install plotly"
    try:
        import imageio
    except ImportError:
        return "Dash installed but ImageIO not available, 3D view will not work. You can fix this by running  $ pip install imageio"
    try:
        dash.Dash('Test')
    except BaseException as e:
        return f"Dash runtime error: {e}"
    return 'OK'


@contextmanager
def plot_solves():
    """
    While `plot_solves()` is active, certain performance optimizations and algorithm implementations may be disabled.
    """
    from . import math
    import pylab
    cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']
    with math.SolveTape(record_trajectories=True) as solves:
        try:
            yield solves
        finally:
            for i, result in enumerate(solves):
                assert isinstance(result, math.SolveInfo)
                from phi.math._tensors import disassemble_tree
                _, (residual,) = disassemble_tree(result.residual)
                residual_mean = math.mean(math.abs(residual), residual.shape.without('trajectory'))
                residual_max = math.max(math.abs(residual), residual.shape.without('trajectory'))
                pylab.plot(residual_mean.numpy(), label=f"{i}: {result.method}", color=cycle[i % len(cycle)])
                pylab.plot(residual_max.numpy(), alpha=0.2, color=cycle[i % len(cycle)])
                print(f"Solve {i}: {result.method} ({1000 * result.solve_time:.1f} ms)\n"
                      f"\t{result.solve}\n"
                      f"\t{result.msg}\n"
                      f"\tConverged: {result.converged}\n"
                      f"\tDiverged: {result.diverged}\n"
                      f"\tIterations: {result.iterations}\n"
                      f"\tFunction evaulations: {result.function_evaluations.trajectory[-1]}")
            pylab.yscale('log')
            pylab.ylabel("Mean / Max Residual")
            pylab.xlabel("Iteration")
            pylab.title(f"Solve Convergence")
            pylab.legend(loc='upper right')
            pylab.savefig(f"pressure-solvers-FP32.png")
            pylab.show()


def count_tensors_in_memory(min_print_size: int = None):
    import sys
    import gc
    from .math import Tensor

    gc.collect()
    total = 0
    bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor):
                total += 1
                size = obj.shape.volume * obj.dtype.itemsize
                bytes += size
                if isinstance(min_print_size, int) and size >= min_print_size:
                    print(f"Tensor '{obj}' ({sys.getrefcount(obj)} references)")
                    # referrers = gc.get_referrers(obj)
                    # print([type(r) for r in referrers])
        except Exception:
            pass
    print(f"There are {total} Φ-Tensors with a total size of {bytes / 1024 / 1024:.1f} MB")
