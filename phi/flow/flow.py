from phi.domain import *
from phi.math.container import *


class State(TensorContainer):

    def __mul__(self, operation):
        return operation(self)


class Physics(object):

    def __init__(self, world=world, dt=1.0):
        """
A Physics object describes a set of physical laws that can be used to simulate a system by moving from state to state,
tracing out a trajectory.
The description of the physical systems (e.g. obstacles, boundary conditions) is also included in the Physics object
and the enclosing world.
        :param world: the world this system lives in
        :param dt: simulation time increment
        """
        self.dt = dt
        self.world = world
        world.register_physics(self)

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


class VolumetricPhysics(Physics):

    def __init__(self, domain, world=world, dt=1.0):
        Physics.__init__(self, world, dt)
        self.domain = domain
        # Cache
        world.on_change(lambda *_: self._update_domain())
        self._update_domain()

    def _update_domain(self):
        pass

    @property
    def grid(self):
        return self.domain.grid

    @property
    def dimensions(self):
        return self.grid.dimensions

    @property
    def rank(self):
        return self.grid.rank






# class Fluid(object):
#
#     def __init__(self, domain=Open3D, gravity=-9.81):
#         self.domain = domain
#         self.gravity = gravity
#         if use_particles:
#             self._particles = np.zeros([1, 0, self.rank])
#         else:
#             self._particles = None
#         # self._density = None  # tensor or PointCloud or None (None = space-filling)
#         # self._velocity = 0  # StaggeredGrid or SampledField or 1D array or 0 (1D/None = global velocity)
#         # self._grids = []  # grid properties: list of tensor / StaggeredGrid
#         # self._sampled = []  # particle properties: list of SampledField
#         self._properties = {}
#
#         self.particle_systems = []
#
#     @property
#     def velocity(self):
#         # always returns a StaggeredGrid or SampledField
#         if not isinstance(self._velocity, (nd.StaggeredGrid, SampledFluidProperty)):
#             raise NotImplementedError()
#         return self._properties["velocity"]
#
#     @velocity.setter
#     def velocity(self, v):
#         self._properties["velocity"] = v
#
#     @property
#     def density(self):
#         return self._properties["density"]
#
#     def advect(self):
#         pass
#
#     def occupied_mask(self):
#         return None
#
#
# class FluidProperty(object):
#
#     def __init__(self):
#         self.conserve_mass = False
#
#     def to_grid(self):
#         raise NotImplementedError()
#
#
# class SampledFluidProperty(FluidProperty):
#
#     def __init__(self, fluid, values, constant_value=np.nan, name=None):
#         FluidProperty.__init__(self)
#         self.fluid = fluid
#         self.values = values  # tensor of shape (batches, particle count, components) or constant value
#         self.constant_value = constant_value  # value where no points exist or nan if only extrapolated
#         self.name = name
#
#     def to_grid(self, griddef):
#         indices = (self.fluid.points - 0.5).astype(np.int)[..., ::-1]
#         indices = math.unstack(indices, axis=-1)
#         array = griddef.zeros(self.values.shape[-1])
#         array[[0] + indices + [slice(None)]] += self.values
#         return array
#
#         # import tensorflow as tf
#         # particles_per_cell = tf.tensor_scatter_add(tf.zeros(), particle_grid_indices, 1)
#         # total_value_per_cell = tf.tensor_scatter_add(tf.zeros(), particle_grid_indices, self.values)
#         # avg_value_per_cell = total_value_per_cell / particles_per_cell
#         # return avg_value_per_cell
#         # alternatively this could use tf.unsorted_segment_sum
#
#     def __add__(self, other):
#         return SampledFluidProperty(self.fluid, self.values+other, self.constant_value)
#
#
# class GridFluidProperty(FluidProperty):
#
#     def __init__(self, conserved=False):
#         FluidProperty.__init__(self)
#         pass

