from phi.tf.flow import *
import time
from functools import partial

# Simulation parameters
k0 = 0.15  # smallest wavenumber in the box
x = 128  # x size
y = 128  # y size
dt = control(0.05)  # timestep
scale = 1 / 100
# Physical Parameters
c1 = 0.1  # adiabatic coefficient [0, None]
# Numerical Parameters
arakawa_coeff = 1  # Poisson bracket coefficient
kappa_coeff = 1  # background flow dy coefficient
nu = 0.0005  # coefficient of hyperdiffusion
N = 3  # laplace**(2*N) diffusion
# Derived
L = 2 * np.pi / k0  # Box Size
dx = L / x  # Grid Spacing
nu = (-1) ** (N + 1) * nu  # Smoothing coefficient & sign
# Packing
PARAMS = dict(c1=c1, nu=nu, N=N, arak=arakawa_coeff, kappa=kappa_coeff)


class Namespace(dict):
    def __mul__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] * val for key, val in self.items()})
        else:
            return Namespace({key: other * val for key, val in self.items()})

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: val / other[key] for key, val in self.items()})
        else:
            return Namespace({key: val / other for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: val / other[key] for key, val in self.items()})
        else:
            return Namespace({key: val / other for key, val in self.items()})

    def __rdiv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] / val for key, val in self.items()})
        else:
            return Namespace({key: other / val for key, val in self.items()})

    def __add__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key: other[key] + val for key, val in self.items()})
        else:
            return Namespace({key: other + val for key, val in self.items()})

    __radd__ = __add__

    def __sub__(self, other):
        return Namespace({key: other - val for key, val in self.items()})

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    @property
    def dtype(self):
        return self["density"].dtype

    def copy(self):
        return Namespace({key: val for key, val in self.items()})


def get_phi(plasma, guess=None):
    """Fourier Poisson Solve for Phi"""
    centered_omega = plasma.omega  # - math.mean(plasma.omega)
    phi = math.fourier_poisson(centered_omega.values, plasma.dx)
    # phi = math.solve_linear(
    #     math.laplace, plasma.omega.values, guess, math.LinearSolve, callback=None
    # )
    return CenteredGrid(
        phi, bounds=plasma.omega.bounds, extrapolation=plasma.omega.extrapolation,
    )  # plasma.omega.domain.grid(phi)


# Diffusion function
def diffuse(arr, N, dx):
    if not isinstance(N, int):
        print(f"{N} {type(N)}")
    for i in range(int(N)):
        arr = field.laplace(arr)  # math.fourier_laplace(arr, dx)
    return arr


def step_gradient_2d(plasma, phi, N=0, nu=0, c1=0, arak=0, kappa=0, dt=0):
    """time gradient of model"""
    # Calculate Gradients
    grad_phi = field.spatial_gradient(phi, stack_dim="gradient")
    dx_p, dy_p = grad_phi.values.gradient.unstack_spatial("x,y")
    # Get difference
    diff = phi - plasma.density
    # Step 2.1: New Omega.
    o = c1 * diff
    if arak:
        o += -arak * math._nd._periodic_2d_arakawa_poisson_bracket(
            phi.values, plasma.omega.values, plasma.dx  # TODO: Fix dx
        )
    if nu and N:
        o += nu * diffuse(plasma.omega, N=N, dx=plasma.dx)
    # Step 2.2: New Density.
    n = c1 * diff
    if arak:
        n += -arak * math._nd._periodic_2d_arakawa_poisson_bracket(
            phi.values, plasma.density.values, plasma.dx
        )
    if kappa:
        n += -kappa * dy_p
    if nu:
        n += nu * diffuse(plasma.density, N=N, dx=plasma.dx)
    return Namespace(
        density=n,
        omega=o,
        phi=phi,  # NOTE: NOT A GRADIENT
        age=plasma.age + dt,
        dx=plasma.dx,
    )


def rk4_step(dt, physics_params, gradient_func=step_gradient_2d, **kwargs):
    gradient_func = partial(gradient_func, **physics_params)
    yn = Namespace(**kwargs)  # given dict to Namespace
    in_age = yn.age
    # Only in the first iteration recalculate phi
    if yn.age == 0:
        pn = get_phi(yn, guess=yn.phi)
    else:
        pn = yn.phi
    k1 = dt * gradient_func(yn, pn, dt=0)
    p1 = get_phi(yn + k1 * 0.5)  # , guess=pn)
    k2 = dt * gradient_func(yn + k1 * 0.5, p1, dt=dt / 2)
    p2 = get_phi(yn + k2 * 0.5)  # , guess=pn+p1*0.5)
    k3 = dt * gradient_func(yn + k2 * 0.5, p2, dt=dt / 2)
    p3 = get_phi(yn + k3)  # , guess=pn+p2*0.5)
    k4 = dt * gradient_func(yn + k3, p3, dt=dt)
    y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    phi = get_phi(y1)  # , guess=pn+p3*0.5)
    return Namespace(
        density=y1.density,
        omega=y1.omega,
        phi=phi,  # TODO: Somehow this does not work properly
        age=in_age + dt,  # y1 contains 2 time steps from compute
        dx=yn.dx,
    )


domain = Domain(x=x, y=y, boundaries=PERIODIC, bounds=Box[0:L, 0:L])
density = domain.grid(math.random_normal(x=x, y=y)) * scale
omega = domain.grid(math.random_normal(x=x, y=y)) * scale
phi = domain.grid(math.random_normal(x=x, y=y)) * scale
age = 0
rk4 = partial(rk4_step, physics_params=PARAMS)
print(
    "\n".join(
        [
            f"x,y:   {x}x{y}",
            f"L:     {L}",
            f"c1:    {c1}",
            f"dt:    {dt}",
            f"N:     {N}",
            f"nu:    {nu}",
            f"scale: {scale}",
        ]
    )
)

for _ in view(density, omega, phi, play=False, framerate=10).range():
    new_state = rk4(dt, density=density, omega=omega, phi=phi, age=age, dx=dx)
    density, omega, phi = new_state["density"], new_state["omega"], new_state["phi"]
    age += dt
