from .volumetric import *
from phi.math import *
from operator import itemgetter


class SmokeState(State):
    __struct__ = StructInfo(('_density', '_velocity'))

    def __init__(self, density, velocity):
        State.__init__(self, tags=('smoke',))
        self._density = density
        self._velocity = velocity if isinstance(velocity, StaggeredGrid) else StaggeredGrid(velocity)

    @property
    def density(self):
        return self._density

    @property
    def velocity(self):
        return self._velocity

    def __eq__(self, other):
        if isinstance(other, SmokeState):
            return self._density == other._density and self._velocity == other._velocity
        else:
            return False

    def __hash__(self):
        return hash(self._density) + hash(self._velocity)

    def __repr__(self):
        return "SmokeState[density: %s, velocity: %s]" % (self.density, self.velocity)

    def __add__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copy(velocity=self.velocity + other)
        else:
            return self.copy(density=self.density+other)

    def __sub__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copy(velocity=self.velocity - other)
        else:
            return self.copy(density=self.density - other)


class Smoke(VolumetricPhysics):

    def __init__(self, domain=Open2D, gravity=-9.81, buoyancy_factor=0.1, conserve_density=False, pressure_solver=None):
        VolumetricPhysics.__init__(self, domain)
        if isinstance(gravity, (tuple, list)):
            assert len(gravity == domain.rank)
            self.gravity = np.array(gravity)
        else:
            gravity = [gravity] + ([0] * (domain.rank - 1))
            self.gravity = np.array(gravity)
        self.buoyancy_factor = buoyancy_factor
        self.conserve_density = conserve_density
        # Pressure Solver
        self.pressure_solver = pressure_solver
        if self.pressure_solver is None:
            from phi.solver.sparse import SparseCG
            self.pressure_solver = SparseCG(accuracy=1e-3)

    def shape(self, batch_size=1):
        return SmokeState(self.grid.shape(batch_size=batch_size), self.grid.staggered_shape(batch_size=batch_size))

    def step(self, smokestate):
        return smokestate * self.advect * self.inflow * self.buoyancy * self.stick * self.divergence_free

    def serialize_to_dict(self):
        return {
            'type': 'Smoke',
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'rank': self.domain.rank,
            'domain': self.domain.serialize_to_dict(),
            'gravity': list(self.gravity),
            'buoyancy_factor': self.buoyancy_factor,
            'conserve_density': self.conserve_density,
            'solver': self.pressure_solver.name,
        }

    def unserialize_from_dict(self):
        raise NotImplementedError()

    def inflow(self, smokestate):
        inflow = inflow_mask(self.worldstate, self.domain.grid)
        return SmokeState(smokestate.density + inflow * self.dt, smokestate.velocity)

    def buoyancy(self, smokestate):
        dv = StaggeredGrid.from_scalar(smokestate.density, self.gravity * self.buoyancy_factor * (-1) * self.dt)
        return SmokeState(smokestate.density, smokestate.velocity + dv)

    def advect(self, smokestate):
        prev_density = smokestate.density
        density = smokestate.velocity.advect(smokestate.density, dt=self.dt)
        velocity = smokestate.velocity.advect(smokestate.velocity, dt=self.dt)
        if self.conserve_density:
            density = nd.normalize_to(density, prev_density)
        return SmokeState(density, velocity)

    def stick(self, smokestate):
        velocity = self.domainstate.with_hard_boundary_conditions(smokestate.velocity)
        # TODO wall friction
        # self.world.geom
        # friction = material.friction_multiplier(dt)
        return SmokeState(smokestate.density, velocity)

    def solve_pressure(self, input):
            """
    Calculates the pressure from the given velocity or velocity divergence using the specified solver.
            :param input: tensor containing the centered velocity divergence values or velocity as StaggeredGrid
            :param solver: PressureSolver to use, options DEFAULT, SCIPY or MANTA
            :return: scalar pressure channel as tensor
            """
            if isinstance(input, StaggeredGrid):
                input = input.divergence()
            if input.shape[-1] == self.domain.rank:
                input = nd.divergence(input, difference='central')

            pressure, iter = self.pressure_solver.solve(input, self.domainstate, pressure_guess=None)
            self.last_pressure, self.last_iter = pressure, iter
            return pressure

    def divergence_free(self, smokestate):
        velocity = self.domainstate.with_hard_boundary_conditions(smokestate.velocity)
        pressure = self.solve_pressure(velocity)
        gradp = StaggeredGrid.gradient(pressure)
        velocity -= self.domainstate.with_hard_boundary_conditions(gradp)
        return SmokeState(smokestate.density, velocity)

