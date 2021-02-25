# Testing Poisson Solvers
from unittest import TestCase
import numpy as np
from phi.flow import *
import matplotlib.pyplot as plt
from functools import partial


# NumPy
def FFT_solve_numpy(tensor, dx, times=1):
    """ Inverse operation to `fourier_laplace`. """
    tensor = tensor.reshape(1, *tensor.shape, 1)
    frequencies = np.fft.fft2(to_complex(tensor), axes=[1, 2])
    k = fftfreq(np.shape(tensor)[1:-1], mode="square")
    fft_laplace = -((2 * np.pi) ** 2) * k
    fft_laplace[(0,) * len(k.shape)] = np.inf
    result = np.real(
        np.fft.ifft2(divide_no_nan(frequencies, fft_laplace ** times), axes=[1, 2])
    ).astype(tensor.dtype)[0, ..., 0]
    return result * dx ** 2


def divide_no_nan(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = x / y
    return np.where(y == 0, 0, result)


def to_complex(x):
    x = np.array(x)
    if x.dtype in (np.complex64, np.complex128):
        return x
    elif x.dtype == np.float64:
        return x.astype(np.complex128)
    else:
        return x.astype(np.complex64)


def fftfreq(resolution, mode="vector", dtype=None):
    assert mode in ("vector", "absolute", "square")
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution], indexing="ij")
    k = expand_dims(np.stack(k, -1), 0)
    k = k.astype(float)
    if mode == "vector":
        return k
    k = np.sum(k ** 2, axis=-1, keepdims=True)
    if mode == "square":
        return k
    else:
        return np.sqrt(k)


def expand_dims(a, axis=0, number=1):
    for _i in range(number):
        a = np.expand_dims(a, axis)
    return a


def ndims(tensor):
    return len(np.shape(tensor))


# PhiFlow
def FFT_solve(*args, **kwargs):
    return math.fourier_poisson(*args, **kwargs)


def CG_solve(grid, guess, dx, padding, **kwargs):
    # guess = guess if guess is not None else domain.grid(0)
    laplace = partial(math.laplace, dx=dx)
    converged, result, iterations = math.solve(
        laplace, grid, guess, math.LinearSolve("CG", **kwargs), callback=None
    )
    print(converged, iterations)
    return result


def CG2_solve(div, guess, **kwargs):
    print(type(div))
    print(type(guess))
    converged, result, iterations = field.solve(
        math.laplace, div, guess, math.LinearSolve(None, **kwargs)
    )
    return result


# Comparison functions
def compare(reference, dics, plot=True, fnc_list=[np.min, np.mean, np.max]):
    count = len(dics)
    diffs = {key: reference - val for key, val in dics.items()}
    heads = " | ".join([f"{'name':<10}"] + [f"{fnc.__name__:>10}" for fnc in fnc_list])
    descs = {
        f"{key:<10}": " | ".join([f"{fnc(val):>10.4g}" for fnc in fnc_list])
        for key, val in diffs.items()
    }
    # Print
    print()
    print(heads)
    for key, val in descs.items():
        print(f"{key:<10} | " + val)
    if plot:
        fig, axarr = plt.subplots(2, 1 + count, figsize=(4.1 * count, 6))
        # Upper Row: results
        im = axarr[0, 0].imshow(reference)
        plt.colorbar(im, ax=axarr[0, 0])
        axarr[0, 0].set_title("reference")
        for i, keyvalues in enumerate(dics.items()):
            im = axarr[0, i + 1].imshow(keyvalues[1])
            plt.colorbar(im, ax=axarr[0, i + 1])
            axarr[0, i + 1].set_title(keyvalues[0])
        im = axarr[1, 0].imshow(np.zeros_like(reference))
        plt.colorbar(im, ax=axarr[1, 0])
        axarr[1, 0].set_title("reference" + " diff")
        for i, keyvalues in enumerate(diffs.items()):
            im = axarr[1, i + 1].imshow(keyvalues[1])
            plt.colorbar(im, ax=axarr[1, i + 1])
            axarr[1, i + 1].set_title(keyvalues[0] + " diff")
        plt.show()


def attempt(fnc):
    try:
        return fnc()
    except:
        return ""


def debug(dics):
    print(
        "\n".join(
            [
                f"{name}: {type(val)}" + attempt(f" ({val.shape})")
                for name, val in dics.items()
            ]
        )
    )


# Test functions
def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / grid_size[0]  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(x + 1) * np.sin(y + 1)
    return d


class TestPoissonSolvers(TestCase):

    def test_poisson(self):
        with math.precision(64):
            steps = 2
            # Physical parameters
            L = 2 * 2 * np.pi
            # Numerical Parameters
            x = 128
            y = 128
            # Derived Parameters
            dx = L / x  # NOTE: 1D
            k0 = 2 * np.pi / L
            # Get input data
            rnd_noise = np.random.rand(x * y).reshape(x, y)
            sine = get_2d_sine((x, y), L)
            # Define
            init_values = sine  # rnd_noise
            domain = Domain(x=x, y=y, boundaries=PERIODIC, bounds=Box[0:L, 0:L])
            sine_grid = domain.grid(math.tensor(init_values, names=["x", "y"]))
            reference = FFT_solve_numpy(sine_grid.values.numpy(order="z,y,x")[0], dx)
            solver_dict = {
                "FFT_solve": lambda x: domain.grid(FFT_solve(x.values, dx)).values.numpy(
                    order="z,y,x"
                )[0],
                "CG_solve": lambda x: CG_solve(
                    x.values,
                    guess=domain.grid(0).values,
                    dx=dx ** 2,
                    padding=PERIODIC,
                    relative_tolerance=1,
                    absolute_tolerance=1e-10,
                    max_iterations=20000,
                ).numpy(order="z,y,x")[0],
                # "CG2_solve": lambda x: CG2_solve(
                #    domain.grid(x.values),
                #    guess=domain.grid(x.values),
                #    accuracy=1e-5,
                #    max_iterations=1000,
                # ),
            }
            solver_soln = {name: fnc(sine_grid) for name, fnc in solver_dict.items()}
            debug(solver_soln)
            compare(reference, solver_soln)
