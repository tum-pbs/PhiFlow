import warnings

from phi import math, struct, field
from phi.field import CenteredGrid, StaggeredGrid
from . import advect
from ._effect import Gravity, effect_applied, gravity_tensor
from ._physics import Physics, StateDependency, State
from .fluid import make_incompressible


@struct.definition()
class Fluid(State):
    """Deprecated. A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid)."""

    def __init__(self, domain, density=0.0, velocity=0.0, buoyancy_factor=0.0, tags=('fluid', 'velocityfield', 'velocity'), name='fluid', **kwargs):
        warnings.warn('Fluid is deprecated. Use a dictionary of fields instead. See the Φ-Flow 2 upgrade instructions.', DeprecationWarning)
        State.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return IncompressibleFlow()

    @struct.constant()
    def domain(self, domain):
        return domain

    @property
    def rank(self):
        return self.domain.rank

    @struct.variable(default=0, dependencies='domain')
    def density(self, density) -> CenteredGrid:
        """
        The marker density is stored in a CenteredGrid with dimensions matching the domain.
        It describes the number of particles per physical volume.

        Args:
          density: 

        Returns:

        """
        return self.domain.grid(density)

    @struct.variable(default=0, dependencies='domain')
    def velocity(self, velocity) -> StaggeredGrid:
        """
        The velocity is stored in a StaggeredGrid with dimensions matching the domain.

        Args:
          velocity: 

        Returns:

        """
        return self.domain.staggered_grid(velocity)

    @struct.constant(default=0.0)
    def buoyancy_factor(self, fac):
        """
        The default fluid physics can apply Boussinesq buoyancy as an upward force, proportional to the density.
        This force is scaled with the buoyancy_factor (float).

        Args:
          fac: 

        Returns:

        """
        return fac

    @struct.variable(default={}, holds_data=False)
    def solve_info(self, solve_info):
        return dict(solve_info)

    def __repr__(self):
        return "Fluid[density: %s, velocity: %s]" % (self.density, self.velocity)


class IncompressibleFlow(Physics):
    """
    Physics modelling the incompressible Navier-Stokes equations.
    Supports buoyancy proportional to the marker density.
    Supports obstacles, density effects, velocity effects, global gravity.

    Args:

    Returns:

    """

    def __init__(self, make_input_divfree=False, make_output_divfree=True, conserve_density=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle', blocking=True),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree
        self.conserve_density = conserve_density

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        density = fluid.density
        pressure, iterations, div = None, None, None
        if self.make_input_divfree:
            velocity, pressure, iterations, div = make_incompressible(velocity, fluid.domain, obstacles)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        if self.conserve_density and fluid.domain.boundaries.accessible_extrapolation == math.extrapolation.ZERO:  # solid boundary
            density = field.normalize(density, fluid.density)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += (density * -gravity * fluid.buoyancy_factor * dt).at(velocity)
        divergent_velocity = velocity
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity, pressure, iterations, div = make_incompressible(velocity, fluid.domain, obstacles)
        solve_info = {
            'pressure': pressure,
            'iterations': iterations,
            'divergence': div,
            'advected_velocity': advected_velocity,
            'divergent_velocity': divergent_velocity,
        }
        return fluid.copied_with(density=density, velocity=velocity, age=fluid.age + dt, solve_info=solve_info)
