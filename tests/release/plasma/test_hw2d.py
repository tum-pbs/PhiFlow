from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
import time

from phi import math
from phi import field
from phi.geom import Box
from phi.math import spatial
from phi.physics._boundaries import Domain, PERIODIC
from .numpy_reference import HW, Namespace
from .phi_version import step_gradient_2d, rk4_step, get_domain_phi

STEPS = 20

with math.precision(64):

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
            [
                str(getattr(element, attr))
                for attr in attr_list
                if hasattr(element, attr)
            ]
        )
        print(output)

    class TestHW2D(TestCase):
        def test_demo_vs_numpy(self):
            dt = 0.1
            # Physical parameters
            L = 4 * 2 * np.pi  # 2 * np.pi / 0.15  # 4*2*np.pi#/0.15
            c1 = 5  # 0.01
            # Numerical Parameters
            grid_pts = 64
            nu = 0  # 1e-8
            N = 0  # 3
            arakawa_coeff = 0
            kappa_coeff = 0
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
            domain = Domain(x=x, y=y, boundaries=PERIODIC, bounds=Box['x,y', 0:L, 0:L])
            hw_state_phi = Namespace(
                density=domain.grid(
                    math.tensor(init_values * density_coeff, spatial('x, y'))
                ),
                omega=domain.grid(
                    math.tensor(init_values * omega_coeff, spatial('x, y'))
                ),
                phi=domain.grid(math.tensor(init_values * phi_coeff, spatial('x, y'))),
                # domain=domain,
                age=0,
                dx=dx,
            )
            from functools import partial

            get_phi = partial(get_domain_phi, domain=domain)
            rk4_step2 = partial(rk4_step, params=params, get_phi=get_phi)
            # Run
            def compare(iterable):
                for k in iterable[0].keys():
                    compare = []
                    for state in iterable:
                        if isinstance(state[k], field._grid.Grid):
                            val = state[k].values.numpy(order="x,y,z")[0]
                        else:
                            val = state[k]
                        compare.append(val)
                    assert len(compare) == 2
                    print(
                        f"  {k:<7}:"
                        + f"  {np.max(np.abs(compare[0]-compare[1])):.7f}"
                        + f"  {np.array_equal(*compare)}"
                        + f"  {np.max(np.abs(compare[0]-compare[1])):.2g}"
                    )
                return True

            np_times = []
            phi_times = []
            for i in range(0, STEPS + 1):
                print(f"step {i:>3} {i*dt:>12.7f}")
                # Numpy
                t0 = time.time()
                hw_state_numpy = hw.rk4_step(hw_state_numpy, dt=dt)
                np_time = time.time() - t0
                np_times.append(np_time)
                # Phi
                t0 = time.time()
                hw_state_phi = rk4_step2(dt=dt, **hw_state_phi)
                phi_time = time.time() - t0
                phi_times.append(phi_time)
                compare([hw_state_numpy, hw_state_phi])
                # if i % 100 == 0:
                #    plot({'numpy': hw_state_numpy.density,
                #          'phi': hw_state_phi.density.values.numpy(order='zyx')[0],
                #          'diff': hw_state_numpy.density - hw_state_phi.density.values.numpy(order='zyx')[0]},
                #         title=f"step: {i}")
                # assert np.allclose(hw_state_numpy.density, hw_state_phi.density.values.numpy(order='zyx')[0])
            print(f"Comparison | NumPy      | PhiFlow")

            def get_str(name, func, vals):
                return " | ".join(
                    [
                        f"{name:<10}",
                        f"{func(vals[0]):<10.4f}",
                        f"{func(vals[1]):<10.4f}",
                    ]
                )

            print(get_str("Mean (s)", np.mean, (np_times, phi_times)))
            print(get_str("Median (s)", np.median, (np_times, phi_times)))
            print(get_str("Min (s)", np.min, (np_times, phi_times)))
            print(get_str("Max (s)", np.max, (np_times, phi_times)))

