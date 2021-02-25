import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse
import scipy.signal
from tqdm import tqdm
from numba import stencil, jit, prange


class Namespace(dict):

    def __mul__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] * val for key, val in self.items()})
        else:
            return Namespace({key:other * val for key, val in self.items()})

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:val / other[key] for key, val in self.items()})
        else:
            return Namespace({key:val / other for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:val / other[key] for key, val in self.items()})
        else:
            return Namespace({key:val / other for key, val in self.items()})

    def __rdiv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] / val for key, val in self.items()})
        else:
            return Namespace({key:other / val for key, val in self.items()})

    def __add__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] + val for key, val in self.items()})
        else:
            return Namespace({key:other + val for key, val in self.items()})

    __radd__ = __add__

    def __sub__(self, other):
        return Namespace({key:other - val for key, val in self.items()})

    @property
    def omega(self):
        return self["omega"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def density(self):
        return self["density"]

    @property
    def age(self):
        return self["age"]

    @property
    def dtype(self):
        return self['density'].dtype

    @property
    def dx(self):
        return self['dx']

    def copy(self):
        return Namespace({key:val for key, val in self.items()})


def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(x + 1) * np.sin(y + 1)
    return d


class HW:

    def __init__(self, N, c1, nu, arakawa_coeff, kappa_coeff, debug=False, quiet=False):
        self.N = N
        self.c1 = c1
        self.nu = (-1)**(self.N + 1) * nu
        self.arakawa_coeff = arakawa_coeff
        self.kappa_coeff = kappa_coeff
        self.debug = debug
        self.test_poisson(size=128)
        self.counter = 0
        if not quiet:
            print(self)

    def __repr__(self):
        rep = "2D Hasegawa-Wakatani Model:\n"
        if self.c1:
            rep += f"[x] c1={self.c1}\n"
        else:
            rep += f"[ ] c1={self.c1}\n"
        if self.nu and self.N:
            rep += f"[x] Diffusion active. N={self.N}, nu={self.nu}\n"
        else:
            rep += f"[ ] Diffusion NOT active. N={self.N}, nu={self.nu}\n"
        if self.arakawa_coeff:
            rep += f"[x] Poisson Bracket included. Coefficient={self.arakawa_coeff}\n"
        else:
            rep += f"[ ] Poisson Bracket NOT incluced. Coefficient={self.arakawa_coeff}\n"
        if self.kappa_coeff:
            rep += f"[x] Background Gradient included. Kappa_coeff={self.kappa_coeff}\n"
        else:
            rep += f"[ ] Background Gradient NOT incluced. Kappa_coeff={self.kappa_coeff}\n"
        return rep

    def euler_step(self, yn, dt):
        pn = self.get_phi(yn, guess=yn.phi)
        k1 = self.step_gradient_2d(yn, pn)
        y1 = yn + dt * k1
        p1 = self.get_phi(y1, guess=yn.phi)
        return Namespace(
            density=y1.density,
            omega=y1.omega,
            phi=p1,
            age=yn.age + dt,  # y1 contains 2 time steps from compute
            dx=yn.dx
        )

    def rk4_step(self, yn, dt=0.1):
        # RK4
        t0 = time.time()
        if yn.age == 0:
            pn = self.get_phi(yn, guess=yn.phi)
        else:
            pn = yn.phi
        k1 = dt * self.step_gradient_2d(yn, pn, dt=0)
        p1 = self.get_phi(yn + k1 * 0.5)  # , guess=pn)
        k2 = dt * self.step_gradient_2d(yn + k1 * 0.5, p1, dt=dt / 2)
        p2 = self.get_phi(yn + k2 * 0.5)  # , guess=pn+p1*0.5)
        k3 = dt * self.step_gradient_2d(yn + k2 * 0.5, p2, dt=dt / 2)
        p3 = self.get_phi(yn + k3)  # , guess=pn+p2*0.5)
        k4 = dt * self.step_gradient_2d(yn + k3, p3, dt=dt)
        y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # phi = #, guess=pn+p3*0.5)
        t1 = time.time()
        if self.debug:
            print(" | ".join([
                f"{yn.age + dt:<7.04g}",
                f"{np.max(np.abs(yn.density.data)):>7.02g}",
                f"{np.max(np.abs(k1.density.data)):>7.02g}",
                f"{np.max(np.abs(k2.density.data)):>7.02g}",
                f"{np.max(np.abs(k3.density.data)):>7.02g}",
                f"{np.max(np.abs(k4.density.data)):>7.02g}",
                f"{t1-t0:>6.02f}s"
            ]))
        return Namespace(
            density=y1.density,
            omega=y1.omega,
            phi=self.get_phi(y1),  # TODO: Somehow this does not work properly
            age=yn.age + dt,  # y1 contains 2 time steps from compute
            dx=yn.dx
        )

    def get_phi(self, plasma, guess=None):
        # Fourier Poisson Solve for Phi
        phi = fourier_poisson(plasma.omega, plasma.dx)
        return phi

    def diffuse(self, arr, N, dx):
        for i in range(N):
            arr = periodic_laplace_func(arr, dx)
        return arr

    def step_gradient_2d(self, plasma, phi, dt=0):
        # Calculate Gradients
        dy_p, dx_p = periodic_gradient(phi, plasma.dx)
        # Get difference
        diff = (phi - plasma.density)
        # Step 2.1: New Omega.
        o = (self.c1 * diff)
        if self.arakawa_coeff:
            o += - self.arakawa_coeff * periodic_arakawa(phi, plasma.omega, plasma.dx)
        if self.nu and self.N:
            o += self.nu * self.diffuse(plasma.omega, self.N, plasma.dx)
        # Step 2.2: New Density.
        n = (self.c1 * diff)
        if self.arakawa_coeff:
            n += - self.arakawa_coeff * periodic_arakawa(phi, plasma.density, plasma.dx)
        if self.kappa_coeff:
            n += - self.kappa_coeff * dy_p
        if self.nu:
            n += self.nu * self.diffuse(plasma.density, self.N, plasma.dx)
        return Namespace(
            density=n,
            omega=o,
            phi=phi,  # NOTE: NOT A GRADIENT
            age=plasma.age + dt,
            dx=plasma.dx
        )

    def test_poisson(self, size=2**8):
        N = size

        def get_2d_sine(grid_size, L):
            indices = np.array(np.meshgrid(*list(map(range, grid_size))))
            phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
            x, y = phys_coord.T
            d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
            return d
        L = 1
        dx = L / N
        sine_field = get_2d_sine((N, N), L=L)
        input_field = 8 * np.pi**2 * sine_field
        reference_result = -sine_field
        #pois_field = poisson_solve(input_field, dx)
        #four_field = fourier_poisson(input_field, dx)
        phi = self.get_phi(Namespace(omega=input_field, phi=input_field, dx=dx), guess=input_field)
        if np.mean(np.abs(reference_result - phi)) < 1e-5:
            pass
        else:
            plot({
                'input': input_field,
                'solved': phi,
                'reference': reference_result,
                'difference': reference_result - phi
            })
            print("! WARNING ! - POISSON SOLVE IS NOT WORKING !")


@stencil
def jpp_nb(zeta, psi, d):
    """dxdy-dydx"""
    return ((zeta[1, 0] - zeta[-1, 0]) * (psi[0, 1] - psi[0, -1])
            - (zeta[0, 1] - zeta[0, -1]) * (psi[1, 0] - psi[-1, 0])) / (4 * d**2)


@stencil
def jpx_nb(zeta, psi, d):
    return (zeta[1, 0] * (psi[1, 1] - psi[1, -1])
            - zeta[-1, 0] * (psi[-1, 1] - psi[-1, -1])
            - zeta[0, 1] * (psi[1, 1] - psi[-1, 1])
            + zeta[0, -1] * (psi[1, -1] - psi[-1, -1])) / (4 * d**2)


@stencil
def jxp_nb(zeta, psi, d):
    return (zeta[1, 1] * (psi[0, 1] - psi[1, 0])
            - zeta[-1, -1] * (psi[-1, 0] - psi[0, -1])
            - zeta[-1, 1] * (psi[0, 1] - psi[-1, 0])
            + zeta[1, -1] * (psi[1, 0] - psi[0, -1])) / (4 * d**2)


@jit  # (nopython=True, parallel=True, nogil=True)
def arakawa_nb(zeta, psi, d):
    return (jpp_nb(zeta, psi, d) + jpx_nb(zeta, psi, d) + jxp_nb(zeta, psi, d)).T / 3


def periodic_arakawa(zeta, psi, d):
    return arakawa_nb(np.pad(zeta, 1, mode='wrap'), np.pad(psi, 1, mode='wrap'), d)[1:-1, 1:-1]


# @jit
def nb_gradient_run(padded, dx):
    fdy = (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    fdx = (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    return fdy, fdx


def periodic_gradient(input_field, dx):
    padded = np.pad(input_field, 1, mode='wrap')
    return nb_gradient_run(padded, dx)


# @jit(nopython=True, nogil=True, parallel=True)
def laplace_np_numba(padded, dx):
    return (
        padded[0:-2, 1:-1]  # above
        + padded[1:-1, 0:-2]  # left
        - 4 * padded[1:-1, 1:-1]  # center
        + padded[1:-1, 2:]  # right
        + padded[2:, 1:-1]  # below
    ) / dx**2


def periodic_laplace_func(a, dx):
    return laplace_np_numba(np.pad(a, 1, 'wrap'), dx)


# @jit(nopython=True, nogil=True, parallel=True)
def grad2d_np_numba(padded, dx):
    return -(
        - padded[0:-2, 1:-1] / 2  # above
        - padded[1:-1, 0:-2] / 2  # left
        + padded[1:-1, 2:] / 2  # right
        + padded[2:, 1:-1] / 2  # below
    ) / dx


def periodic_grad2d_func(a, dx):
    return grad2d_np_numba(np.pad(a, 1, 'wrap'), dx)


def get_energy(n, phi, dx):
    phi_gradients = periodic_grad2d_func(phi, dx)
    return np.sum(n**2 + np.abs(phi_gradients)**2) * dx**2 / 2


def fourier_poisson(tensor, dx, times=1):
    """ Inverse operation to `fourier_laplace`. """
    tensor = tensor.reshape(1, *tensor.shape, 1)
    frequencies = np.fft.fft2(to_complex(tensor), axes=[1, 2])
    k = fftfreq(np.shape(tensor)[1:-1], mode='square')
    fft_laplace = -(2 * np.pi)**2 * k
    fft_laplace[(0,) * len(k.shape)] = np.inf
    result = np.real(np.fft.ifft2(divide_no_nan(frequencies, fft_laplace**times), axes=[1, 2])).astype(tensor.dtype)[0, ..., 0]
    return result * dx**2


def divide_no_nan(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
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


def fftfreq(resolution, mode='vector', dtype=None):
    assert mode in ('vector', 'absolute', 'square')
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution], indexing='ij')
    k = expand_dims(np.stack(k, -1), 0)
    k = k.astype(float)
    if mode == 'vector':
        return k
    k = np.sum(k**2, axis=-1, keepdims=True)
    if mode == 'square':
        return k
    else:
        return np.sqrt(k)


def expand_dims(a, axis=0, number=1):
    for _i in range(number):
        a = np.expand_dims(a, axis)
    return a


def ndims(tensor):
    return len(np.shape(tensor))
