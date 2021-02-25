from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
import time

from phi import math
from phi import field
from phi.app import App
from phi.geom import Box
from phi.physics import Domain, PERIODIC
from .numpy_reference import HW, Namespace
from .phi_version import step_gradient_2d, rk4_step, get_domain_phi

math.set_gloabl_precision(64)


def plot_list(image_arrays):
    N = len(image_arrays)
    fig, axarr = plt.subplots(1, N, figsize=(3.5 * N, 4))
    for i in range(N):
        c = axarr[i].imshow(image_arrays[i])
        plt.colorbar(c, ax=axarr[i])
    return fig


def plot_dict(plot_dic):
    n = len(plot_dic.values())
    fig, axarr = plt.subplots(1, n, figsize=(n * 4, 4))
    for i, it in enumerate(plot_dic.items()):
        label, field = it
        c = axarr[i].imshow(field)
        plt.colorbar(c, ax=axarr[i])
        axarr[i].set_title(label)
    return fig


def plot(data, title=""):
    if isinstance(data, list):
        fig = plot_list(data)
    elif isinstance(data, dict):
        fig = plot_dict(data)
    else:
        raise
    if len(title):
        fig.suptitle(title)
    plt.show()


def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / grid_size[0]  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(x + 1) * np.sin(y + 1)
    return d


def debug(element):
    attr_list = ["dtype", "shape"]
    output = str(type(element))
    output += " " + " ".join(
        [str(getattr(element, attr)) for attr in attr_list if hasattr(element, attr)]
    )
    print(output)


class TestHW2D(TestCase):
    def test_demo_vs_numpy(self):
        steps = 100
        dt = 0.1
        # Physical parameters
        L = 4 * 2 * np.pi  # 2 * np.pi / 0.15  # 4*2*np.pi#/0.15
        c1 = 1  # 0.01
        # Numerical Parameters
        grid_pts = 128
        nu = 1e-8
        N = 3
        arakawa_coeff = 1
        kappa_coeff = 1
        # Derived Parameters
        dx = L / grid_pts
        k0 = 2 * np.pi / L
        # Get input data
        rnd_noise = np.random.rand(grid_pts * grid_pts).reshape(grid_pts, grid_pts)
        sine = get_2d_sine((grid_pts, grid_pts), L)
        init_values = sine / 1000
        density_coeff = 1
        omega_coeff = -1 / 2
        phi_coeff = -1 / 2
        x = grid_pts
        y = grid_pts
        params = dict(
            c1=c1, nu=nu, N=N, arakawa_coeff=arakawa_coeff, kappa_coeff=kappa_coeff
        )
        # NumPy reference
        hw = HW(
            c1=c1,
            nu=nu,
            N=N,
            arakawa_coeff=arakawa_coeff,
            kappa_coeff=kappa_coeff,
            debug=False,
        )
        hw_state_numpy = Namespace(
            density=init_values * density_coeff,
            omega=init_values * omega_coeff,
            phi=init_values * phi_coeff,
            age=0,
            dx=dx,
        )
        # Phi version
        domain = Domain(x=x, y=y, boundaries=PERIODIC, bounds=Box[0:L, 0:L])
        hw_state_phi = Namespace(
            density=domain.grid(
                math.tensor(init_values * density_coeff, names=["x", "y"])
            ),
            omega=domain.grid(math.tensor(init_values * omega_coeff, names=["x", "y"])),
            phi=domain.grid(math.tensor(init_values * phi_coeff, names=["x", "y"])),
            # domain=domain,
            age=0,
            dx=dx,
        )
        from functools import partial

        get_phi = partial(get_domain_phi, domain=domain)
        rk4_step2 = partial(rk4_step, params=params, get_phi=get_phi)
        app = App("Hasegawa-Wakatani", dt=dt)
        app.set_state(
            hw_state_phi, step_function=rk4_step2, show=["density", "omega", "phi"]
        )
        app.prepare()
        # Run
        def compare(iterable):
            for k in iterable[0].keys():
                compare = []
                for state in iterable:
                    if isinstance(state[k], field._grid.Grid):
                        val = state[k].values.numpy(order="zyx")[0]
                    else:
                        val = state[k]
                    compare.append(val)
                assert len(compare) == 2
                print(
                    f"  {k:<7}:  {np.max(np.abs(compare[0]-compare[1])):.7f}  {np.array_equal(*compare)} {np.max(np.abs(compare[0]-compare[1])):.2g}"
                )
            return True

        np_times = []
        phi_times = []
        for i in range(0, steps + 1):
            print(f"step {i:>3} {1e-5:>12.7f}")
            # Numpy
            t0 = time.time()
            hw_state_numpy = hw.rk4_step(hw_state_numpy, dt=dt)
            np_time = time.time() - t0
            np_times.append(np_time)
            # Phi
            t0 = time.time()
            app.step()
            phi_time = time.time() - t0
            phi_times.append(phi_time)
            hw_state_phi = app.state
            compare([hw_state_numpy, hw_state_phi])
            # if i % 100 == 0:
            #    plot({'numpy': hw_state_numpy.density,
            #          'phi': hw_state_phi.density.values.numpy(order='zyx')[0],
            #          'diff': hw_state_numpy.density - hw_state_phi.density.values.numpy(order='zyx')[0]},
            #         title=f"step: {i}")
            # assert np.allclose(hw_state_numpy.density, hw_state_phi.density.values.numpy(order='zyx')[0])
        print(f"Comparison | NumPy      | PhiFlow")
        print(f"Mean (s)   | {np.mean(np_times):<10.4f} | {np.mean(phi_times):<10.4f}")
        print(
            f"Median (s) | {np.median(np_times):<10.4f} | {np.median(phi_times):<10.4f}"
        )
        print(f"Min (s)    | {np.min(np_times):<10.4f} | {np.min(phi_times):<10.4f}")
        print(f"Max (s)    | {np.max(np_times):<10.4f} | {np.max(phi_times):<10.4f}")
