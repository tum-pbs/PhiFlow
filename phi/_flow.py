from phi.math.nd import *
import numbers, six, threading, contextlib

from phi.geom import *


def advect(field, velocity, scene, dt=1, interpolation="LINEAR"):
    if isinstance(velocity, StaggeredGrid):
        return velocity.advect(field, interpolation=interpolation, dt=dt)
    idx = indices_tensor(velocity)
    velocity = velocity[..., ::-1]
    sample_coords = idx - velocity * dt
    result = math.resample(field, sample_coords, interpolation=interpolation, boundary="REPLICATE")
    return result






class _PhiStack(threading.local):

    def __init__(self):
        self.stack = []

    def get_default(self, raise_error=True):
        if raise_error:
            assert len(self.stack) > 0, "Default simulation required. Use 'with simulation:' or 'with simulation.as_default():"
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @contextlib.contextmanager
    def get_controller(self, default):
        """Returns a context manager for manipulating a default stack."""
        try:
            self.stack.append(default)
            yield default
        finally:
            # stack may be empty if reset() was called
            if self.stack:
                self.stack.remove(default)


_default_phi_stack = _PhiStack()


class FluidSimulation(object):
    def __init__(self, shape, boundary="closed",
                 batch_size=1, gravity=-9.81, buoyancy_factor=0.01,
                 solver=None, force_use_masks=False, single_domain=True, sampling="mac"):
        """

        :param shape: List or tuple describing the dimensions of the simulation in the order [z, y, x]
        :param batch_size: the default batch size that is used for all created fields unless otherwise specified.
        If None, TensorFlow tensors will have an undefined batch size and NumPy arrays will have a batch size of 1.
        :param gravity: Single value or 1D array of same length as shape
        :param buoyancy_factor: Single value
        """
        self._dimensions = [d for d in shape]
        self._dt = 1
        self._dx = 1.0 / max(shape)
        if isinstance(gravity, float):
            self._gravity = [0.] * len(shape)
            self._gravity[self.up_dim] = gravity
        else:
            assert isinstance(gravity, tuple) or isinstance(gravity, list)
            assert len(
                gravity) == self.rank, "Entries in gravity must correspond to number of dimensions, got %d" % len(
                gravity)
            self._gravity = list(gravity)
        self._batch_size = batch_size
        self._buoyancy_factor = buoyancy_factor
        self._boundary_velocity = None
        self._sticky_walls = False
        self._boundary = None
        if isinstance(boundary, six.string_types):
            if boundary == "open":
                boundary = DomainBoundary(True)
            elif boundary == "closed":
                boundary = DomainBoundary(False)
            else:
                raise ValueError("Illegal boundary: %s" % boundary)
        self.boundary = boundary  # checks rank
        assert sampling in ("mac", "center")
        self._mac = sampling == "mac"
        self._single_domain = single_domain
        self._force_use_masks = force_use_masks
        self._fluid_mask = None  # can be None
        self._active_mask = None  # can be None
        self._velocity_mask = None  # can be None
        self.clear_domain()
        self._default_simulation_context_manager = None
        self.last_pressure, self.last_iter = None, None
        if solver is None:
            self._solver = self._build_default_solver()
        else:
            self._solver = solver

    def _build_default_solver(self):
        from phi.solver.sparse import SparseCG
        if max(self._dimensions) < 80:
            return SparseCG(accuracy=1e-4)
        else:
            grid_count = int(max(0, np.ceil(np.log2(max(self._dimensions) / 80.0)))) + 1
            solvers = [SparseCG(1e-4 * 2**i) for i in range(grid_count)]
            from phi.solver.multigrid import Multigrid
            return Multigrid(solvers)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def rank(self):
        return len(self._dimensions)

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if not isinstance(boundary, DomainBoundary):
            raise ValueError("Boundaries must be an instance of DomainBoundary but is %s" % type(boundary))
        boundary.check_rank(self.rank)
        self._boundary = boundary

    @property
    def solver(self):
        return self._solver

    @property
    def up_dim(self):
        if self.rank == 0:
            return None
        elif self.rank == 1:
            return 0
        else:
            return self.rank - 2

    @property
    def extended_fluid_mask(self):
        if self._fluid_mask is None:
            return self._boundary.pad_fluid(self._create_mask())
        return self._boundary.pad_fluid(self._fluid_mask)

    @property
    def extended_active_mask(self):
        if self._active_mask is None:
            return self._boundary.pad_active(self._create_mask())
        return self._boundary.pad_active(self._active_mask)

    def as_default(self):
        return _default_phi_stack.get_controller(self)

    def __enter__(self):
        if self._default_simulation_context_manager is None:
            self._default_simulation_context_manager = self.as_default()
        return self._default_simulation_context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._default_simulation_context_manager.__exit__(exc_type, exc_val, exc_tb)
        self._default_simulation_context_manager = None

    def clear_domain(self):
        """
Removes all obstacles and empty cells from the simulation domain.
        """
        if not self._force_use_masks:
            self._active_mask = None
            self._fluid_mask = None
        else:
            self._active_mask = self._create_mask()
            self._fluid_mask = self._create_mask()
        self._velocity_mask = self._boundary.create_velocity_mask(self._fluid_mask, self._dimensions, self._mac)

    def set_obstacle(self, mask_or_size, origin=None):
        if self._active_mask is None:
            self._active_mask = self._create_mask()
        if self._fluid_mask is None:
            self._fluid_mask = self._create_mask()

        dims = range(self.rank)

        if isinstance(mask_or_size, np.ndarray):
            value = mask_or_size
            slices = None
            raise NotImplementedError()  # TODO
        else:
            # mask_or_size = tuple/list of extents
            if isinstance(mask_or_size, int):
                mask_or_size = [mask_or_size for i in dims]
            if origin is None:
                origin = [0 for i in range(len(mask_or_size))]
            else:
                origin = list(origin)
            self._fluid_mask[[0] + [slice(origin[i], origin[i] + mask_or_size[i]) for i in dims] + [0]] = 0
            self._active_mask[[0] + [slice(origin[i], origin[i] + mask_or_size[i]) for i in dims] + [0]] = 0
        self._velocity_mask = self._boundary.create_velocity_mask(self._fluid_mask, self._dimensions, self._mac)

    def _create_mask(self):
        if self._single_domain:
            return self.ones("scalar", 1, np.int8)
        else:
            return self.ones("scalar", None, np.int8)

    def shape(self, element_type="scalar", batch_size=None, scale=None):
        """
Returns the shape including batch dimension and component dimension of a tensor containing the given element type.
This shape corresponds to the dimensionality of tensors used in the simulation.
The shape of centered fields is (batch size, spatial dimensions..., component size).
For staggered (MAC) grids, each spatial dimension is increased by 1.
        :param element_type: Vector length as int or one of ("scalar", "velocity", "staggered", "vector") (default "scalar")
        :param batch_size: batch dimension of array. If None, the default batch size of the simulation is used.
        :param scale: The simulation dimensions are first scaled by this factor and cast to an integer.
        :return: The shape as a tuple or list
        """
        if batch_size is None: batch_size = self._batch_size

        if not scale:
            dimensions = self.dimensions
        else:
            dimensions = [int(d * scale) for d in self.dimensions]

        if element_type == "velocity":
            element_type = "staggered" if self._mac else "vector"

        if isinstance(element_type, numbers.Number):
            return [batch_size] + dimensions + [element_type]
        if element_type == "scalar":
            return [batch_size] + dimensions + [1]
        if element_type == "staggered":
            return [batch_size] + [d + 1 for d in dimensions] + [len(dimensions)]
        if element_type == "vector":
            return [batch_size] + dimensions + [len(dimensions)]
        raise ValueError("Illegal element type {}".format(element_type))

    def _wrap(self, field, element_type):
        if element_type == "velocity":
            element_type = "staggered" if self._mac else "vector"
        if element_type == "staggered":
            return StaggeredGrid(field)
        else:
            return field

    def zeros(self, element_type=1, batch_size=None, dtype=np.float32):
        """
Creates a NumPy array of zeros of which the shape is determined by the dimensions of the simulation, the given element
type and batch size.
The shape of the array corresponds to the result of :func:`shape`.
If the element type is 'mac', an instanceof StaggeredGrid holding the array is returned.
        :param element_type: Vector length as int or one of ("scalar", "staggered", "vector") (default "scalar")
        :param batch_size: batch dimension of array. If None, the default batch size of the simulation is used.
        :param dtype: NumPy data type
        :return: NumPy array of zeros or StaggeredGrid holding the array
        """
        batch_size = batch_size if batch_size is not None else self._batch_size if self._batch_size is not None else 1
        return self._wrap(np.zeros(self.shape(element_type, batch_size), dtype), element_type)

    def ones(self, element_type=1, batch_size=None, dtype=np.float32):
        """
Creates a NumPy array of ones of which the shape is determined by the dimensions of the simulation, the given element
type and batch size.
The shape of the array corresponds to the result of :func:`shape`.
If the element type is 'mac', an instanceof StaggeredGrid holding the array is returned.
        :param element_type: Vector length as int or one of ("scalar", "staggered", "vector") (default "scalar")
        :param batch_size: batch dimension of array. If None, the default batch size of the simulation is used.
        :param dtype: NumPy data type
        :return: NumPy array of zeros or StaggeredGrid holding the array
        """
        batch_size = batch_size if batch_size is not None else self._batch_size if self._batch_size is not None else 1
        return self._wrap(np.ones(self.shape(element_type, batch_size), dtype), element_type)

    def reshape(self, tensor):
        if len(tensor.shape) == len(self.dimensions):
            return math.reshape(tensor, [1] + self.dimensions + [1])
        if len(tensor.shape) == len(self.dimensions) + 1:
            if tensor.shape[-1] == len(self.dimensions) or tensor.shape[-1] == 1:
                return math.reshape(tensor, [1] + self.dimensions + [tensor.shape[-1]])
        raise ValueError("Unsupported")

    def indices(self, centered=False, staggered=False):
        if staggered:
            idx_zxy = np.meshgrid(*[np.arange(-0.5,dim+0.5,1) for dim in self.dimensions][::-1])[::-1]
        else:
            idx_zxy = np.meshgrid(*[range(dim) for dim in self.dimensions][::-1])[::-1]
        if centered:
            idx_zxy = [idx_zxy[i] - self.dimensions[i] // 2 for i in range(len(self.dimensions))]
        return idx_zxy

    def buoyancy(self, density, sampling="default"):
        assert sampling in ("default", "center", "mac")
        mac = self._mac if sampling == "default" else sampling == "mac"
        if mac:
            return StaggeredGrid.from_scalar(density, self._gravity) * self._dt * -self._buoyancy_factor
        else:
            F = self._dt * -self._buoyancy_factor * density[..., 0]
            return math.stack([F * g for g in self._gravity][::-1], axis=-1)

    def with_boundary_conditions(self, velocity):
        if self._velocity_mask is None:
            return velocity
        masked = velocity * self._velocity_mask
        return masked if self._boundary_velocity is None else masked + self._boundary_velocity

    def conserve_mass(self, target, source):
        return normalize_to(target, source)

    def solve_pressure(self, input, solver=None):
        """
Calculates the pressure from the given velocity or velocity divergence using the specified solver.
        :param input: tensor containing the centered velocity divergence values or velocity as StaggeredGrid
        :param solver: PressureSolver to use, options DEFAULT, SCIPY or MANTA
        :return: scalar pressure field as tensor
        """
        if isinstance(input, StaggeredGrid):
            input = input.divergence()
        if input.shape[-1] == len(self.dimensions):
            input = divergence(input, difference="central")

        solver = self._solver if solver is None else solver
        pressure, iter = solver.solve(input, self._active_mask, self._fluid_mask, self._boundary, pressure_guess=None)
        self.last_pressure, self.last_iter = pressure, iter
        return pressure

    def divergence_free(self, velocity, enforce_boundary_conditions=True, **kwargs):
        if enforce_boundary_conditions:
            velocity = self.with_boundary_conditions(velocity)
        pressure = self.solve_pressure(velocity, **kwargs)
        gradp = self.gradient_velocity(pressure)
        if enforce_boundary_conditions:
            velocity -= self.with_boundary_conditions(gradp)
        else:
            velocity -= gradp
        return velocity

        # for particles
        velocity = velocity_field.to_grid()
        active_mask = SampledField(velocity_field.points, 1).to_grid()
        velocity = divergence_free(velocity, active_mask)
        velocity = extrapolate(velocity, 1)
        return sample(velocity, velocity_field.points)

    @property
    def griddef(self):
        return Grid(self.dimensions)

    def gradient_velocity(self, field):
        if self._mac:
            return StaggeredGrid.gradient(field)
        else:
            return gradient(field, difference="central")


    def advect(self, field, velocity):
        # Grids: as before
        # PointCloud by SampledField: shift point cloud (use cache)
        # SampledField by SampledField: shift point cloud (use cache) and return new, rebased field

        # advect given the boundary conditions of the simulation
        assert len(velocity.values.shape[-1]) == len(
            velocity.points.shape) - 2, "Only vector fields can be used to advect"
        raise NotImplementedError()

    def gravity(self):
        raise NotImplementedError()


    def as_dict(self):
        return {
            "dimensions": self.dimensions,
            "rank": self.rank,
            "batch_size": self.batch_size,
            "solver": self.solver.name,
            "open_boundary": self._boundary._open,
            "gravity": self._gravity,
            "buoyancy_factor": self._buoyancy_factor,
        }



def _is_mantatensor_installed():
    try:
        import mantatensor.mantatensor_bindings
        return True
    except:
        return False


def _is_scipy_installed():
    try:
        import scipy
        return True
    except ImportError:
        return False


