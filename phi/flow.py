from phi.domain import *
from phi.math import *


# Possible particle / grid combinations:
# Velocity-grid with velocity-density
# Velocity-grid with particle-density
# particle-velocity with particle-density but grid-pressure (FLIP)

# whenever a particle property is required,


class Simulatable():

    def step(self, *args):
        pass


class ParticleSystem(Simulatable):
    pass

class GridSimulation(Simulatable):
    pass


class Smoke(object):

    def __init__(self, domain=Open2D, world=world, gravity=-9.81, buoyancy=0.1, conserve_density=False, batch_size=None, pressure_solver=None):
        self.domain = domain
        if isinstance(gravity, (tuple, list)):
            assert len(gravity == domain.rank)
            self.gravity = np.array(gravity)
        else:
            gravity = [gravity] + ([0] * (domain.rank - 1))
            self.gravity = np.array(gravity)
        self.batch_size = batch_size
        self.world = world
        self._buoyancy = buoyancy
        self._conserve_density = conserve_density

        self.pressure_solver = pressure_solver
        if self.pressure_solver is None:
            from phi.solver.sparse import SparseCG
            self.pressure_solver = SparseCG(accuracy=1e-3)

        self.density = domain.grid.zeros(1, self.batch_size)
        self.velocity = domain.grid.staggered_zeros(self.batch_size)
        world.on_change(lambda *_: self._update_domain())

    def step(self, dt=1.0):
        return self.with_inflow(dt).buoyancy(dt).friction(dt).advect(dt).divergence_free()

    def _update_domain(self):
        mask = 1 - geometry_mask(self.world, self.domain.grid, 'obstacle')
        new_domain = Domain(self.domain.grid, self.domain.boundaries, active=mask, accessible=mask)
        self.domain = new_domain  # TODO should this object really change?

    def with_inflow(self, dt=1.0):
        inflow = inflow_mask(self.world, self.domain.grid)
        self.density = self.density + inflow * dt
        return self

    def advect(self, dt=1.0):
        prev_density = self.density
        self.density = self.velocity.advect(self.density, dt=dt)
        self.velocity = self.velocity.advect(self.velocity, dt=dt)
        if self._conserve_density:
            self.density = nd.normalize_to(self.density, prev_density)
        return self

    def __add__(self, other):
        assert isinstance(other, Geometry)
        geom = other
        points = self.domain.grid.center_points()
        self.density += geom.value_at(points)
        return self

    def buoyancy(self, dt=1.0):
        self.velocity += StaggeredGrid.from_scalar(self.density, self.gravity * self._buoyancy * (-1) * dt)
        return self

    def friction(self, dt=1.0):
        self.velocity = self.domain.with_hard_boundary_conditions(self.velocity)
        # TODO friction


        # self.world.geom
        # friction = material.friction_multiplier(dt)

        return self

    def solve_pressure(self, input, solver=None):
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

            solver = self.pressure_solver if solver is None else solver
            pressure, iter = solver.solve(input, self.domain, pressure_guess=None)
            self.last_pressure, self.last_iter = pressure, iter
            return pressure

    def divergence_free(self, solver=None):
        velocity = self.domain.with_hard_boundary_conditions(self.velocity)
        pressure = self.solve_pressure(velocity, solver)
        gradp = StaggeredGrid.gradient(pressure)
        velocity -= self.domain.with_hard_boundary_conditions(gradp)
        self.velocity = velocity
        return self



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

