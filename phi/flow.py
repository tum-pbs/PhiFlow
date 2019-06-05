from phi.domain import *
from phi.math import *
from phi.math.container import *


class Simulation(object):

    def __init__(self, world=world, dt=1.0):
        self.dt = dt
        self.world = world
        world.register_simulation(self)

    def step(self, state):
        """
Computes the next state of the simulation, given the current state.
Solves the simulation for a time increment self.dt.
        :param state: current state
        :return next state
        """
        raise NotImplementedError(self)

    def empty(self, batch_size=1):
        """
Creates a new SimState instance that represents an empty / default state of the simulation.
        """
        raise NotImplementedError(self)

    def serialize_to_dict(self):
        raise NotImplementedError(self)

    def unserialize_from_dict(self):
        raise NotImplementedError(self)


class SimState(TensorContainer):

    def __mul__(self, operation):
        return operation(self)


class SmokeState(SimState):

    def __init__(self, density, velocity):
        self._density = density
        self._velocity = velocity

    @property
    def density(self):
        return self._density

    @property
    def velocity(self):
        return self._velocity

    def disassemble(self):
        v, v_re = disassemble(self._velocity)
        return [self._density] + v, lambda tensors: SmokeState(tensors[0], v_re(tensors[1:]))


class Smoke(Simulation):

    def __init__(self, domain=Open2D, world=world, dt=1.0,
                 gravity=-9.81, buoyancy_factor=0.1, conserve_density=False, pressure_solver=None):
        Simulation.__init__(self, world, dt)
        self.domain = domain
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
        # Cache
        world.on_change(lambda *_: self._update_domain())
        self._update_domain()

    @property
    def grid(self):
        return self.domain.grid

    @property
    def dimensions(self):
        return self.grid.dimensions

    def step(self, smokestate):
        return smokestate * self.advect * self.inflow * self.buoyancy * self.friction * self.divergence_free

    def empty(self, batch_size=1):
        density = self.domain.grid.zeros(1, batch_size)
        velocity = self.domain.grid.staggered_zeros(batch_size)
        return SmokeState(density, velocity)

    def _update_domain(self):
        mask = 1 - geometry_mask(self.world, self.domain.grid, 'obstacle')
        self.domainstate = DomainState(self.domain, self.world.state, active=mask, accessible=mask)

    def serialize_to_dict(self):
        return {
            "type": "smoke",
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "rank": self.domain.rank,
            "domain": self.domain.serialize_to_dict(),
            "gravity": list(self.gravity),
            "buoyancy_factor": self.buoyancy_factor,
            "conserve_density": self.conserve_density,
            "solver": self.pressure_solver.name,
        }

    def unserialize_from_dict(self):
        raise NotImplementedError()

    def inflow(self, smokestate):
        inflow = inflow_mask(self.world, self.domain.grid)
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

    def friction(self, smokestate):
        velocity = self.domainstate.with_hard_boundary_conditions(smokestate.velocity)
        # TODO friction
        # self.world.geom
        # friction = material.friction_multiplier(dt)
        return SmokeState(smokestate.density, velocity)

    def solve_pressure(self, input):
            """
    Calculates the pressure from the given velocity or velocity divergence using the specified solver.
            :param input: tensor containing the centered velocity divergence values or velocity as StaggeredGrid
            :param solver: PressureSolver to use, options DEFAULT, SCIPY or MANTA
            :return: scalar pressure field as tensor
            """
            if isinstance(input, StaggeredGrid):
                input = input.divergence()
            if input.shape[-1] == self.domain.rank:
                input = nd.divergence(input, difference="central")

            pressure, iter = self.pressure_solver.solve(input, self.domainstate, pressure_guess=None)
            self.last_pressure, self.last_iter = pressure, iter
            return pressure

    def divergence_free(self, smokestate):
        velocity = self.domainstate.with_hard_boundary_conditions(smokestate.velocity)
        pressure = self.solve_pressure(velocity)
        gradp = StaggeredGrid.gradient(pressure)
        velocity -= self.domainstate.with_hard_boundary_conditions(gradp)
        return SmokeState(smokestate.density, velocity)





class Fluid(object):

    def __init__(self, domain=Open3D, gravity=-9.81):
        self.domain = domain
        self.gravity = gravity
        if use_particles:
            self._particles = np.zeros([1, 0, self.rank])
        else:
            self._particles = None
        # self._density = None  # tensor or PointCloud or None (None = space-filling)
        # self._velocity = 0  # StaggeredGrid or SampledField or 1D array or 0 (1D/None = global velocity)
        # self._grids = []  # grid properties: list of tensor / StaggeredGrid
        # self._sampled = []  # particle properties: list of SampledField
        self._properties = {}

        self.particle_systems = []

    @property
    def velocity(self):
        # always returns a StaggeredGrid or SampledField
        if not isinstance(self._velocity, (nd.StaggeredGrid, SampledFluidProperty)):
            raise NotImplementedError()
        return self._properties["velocity"]

    @velocity.setter
    def velocity(self, v):
        self._properties["velocity"] = v

    @property
    def density(self):
        return self._properties["density"]

    def advect(self):
        pass

    def occupied_mask(self):
        return None


class FluidProperty(object):

    def __init__(self):
        self.conserve_mass = False

    def to_grid(self):
        raise NotImplementedError()


class SampledFluidProperty(FluidProperty):

    def __init__(self, fluid, values, constant_value=np.nan, name=None):
        FluidProperty.__init__(self)
        self.fluid = fluid
        self.values = values  # tensor of shape (batches, particle count, components) or constant value
        self.constant_value = constant_value  # value where no points exist or nan if only extrapolated
        self.name = name

    def to_grid(self, griddef):
        indices = (self.fluid.points - 0.5).astype(np.int)[..., ::-1]
        indices = math.unstack(indices, axis=-1)
        array = griddef.zeros(self.values.shape[-1])
        array[[0] + indices + [slice(None)]] += self.values
        return array

        # import tensorflow as tf
        # particles_per_cell = tf.tensor_scatter_add(tf.zeros(), particle_grid_indices, 1)
        # total_value_per_cell = tf.tensor_scatter_add(tf.zeros(), particle_grid_indices, self.values)
        # avg_value_per_cell = total_value_per_cell / particles_per_cell
        # return avg_value_per_cell
        # alternatively this could use tf.unsorted_segment_sum

    def __add__(self, other):
        return SampledFluidProperty(self.fluid, self.values+other, self.constant_value)


class GridFluidProperty(FluidProperty):

    def __init__(self, conserved=False):
        FluidProperty.__init__(self)
        pass

