from phi import math, struct
import numpy as np


class Geometry(struct.Struct):

    def value_at(self, location):
        raise NotImplementedError(self.__class__)

    def at(self, grid):
        return self.value_at(grid.center_points())


class Box(Geometry):
    __struct__ = struct.Def((), ('origin', 'size'))

    def __init__(self, origin, size):
        self._origin = math.as_tensor(origin)
        self._size = math.as_tensor(size)
        self._upper = self.origin + self.size

    @property
    def origin(self):
        return self._origin

    @property
    def size(self):
        return self._size

    @property
    def upper(self):
        return self._upper

    @property
    def spatial_rank(self):
        return len(self.size)

    def global_to_local(self, global_position):
        return (global_position - self.origin) / self.size

    def local_to_global(self, local_position):
        return local_position * self.size + self.origin

    def value_at(self, global_position):
        # local = self.global_to_local(global_position)
        # bool_inside = (local >= 0) & (local <= 1)
        bool_inside = (global_position >= self.origin) & (global_position <= (self.upper))
        bool_inside = math.all(bool_inside, axis=-1, keepdims=True)
        return math.to_float(bool_inside)

    def contains(self, other):
        if isinstance(other, Box):
            return np.all(other.origin >= self.origin) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()


class BoxGenerator(object):

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        origin = []
        size = []
        for dim in item:
            if isinstance(dim, (int, float)):
                origin.append(dim)
                size.append(1)
            elif isinstance(dim, slice):
                assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
                origin.append(dim.start)
                size.append(dim.stop - dim.start)
        return Box(origin, size)


box = BoxGenerator()


class Sphere(Geometry):
    __struct__ = struct.Def((), ('_center', '_radius'))

    def __init__(self, center, radius):
        self._center = math.as_tensor(center)
        self._radius = math.as_tensor(radius)

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    def value_at(self, location):
        bool_inside = math.expand_dims(math.sum((location - self._center)**2, axis=-1) <= self._radius ** 2, -1)
        return math.to_float(bool_inside)


class Grid(struct.Struct):
    __struct__ = struct.Def((), ('_resolution', '_box'))

    def __init__(self, resolution, box=None):
        self._resolution = np.array(resolution)
        if box is not None:
            self._box = box
        else:
            self._box = Box([0 for d in resolution], self._resolution)

    @property
    def resolution(self):
        return self._resolution

    @property
    def box(self):
        return self._box

    @property
    def rank(self):
        return len(self._resolution)

    def cell_index(self, global_position):
        local_position = self._box.global_to_local(global_position) * self._resolution
        position = math.to_int(local_position - 0.5)
        position = math.maximum(0, position)
        position = math.minimum(position, self._resolution-1)
        return position

    def center_points(self):
        idx_zyx = np.meshgrid(*[np.arange(0.5,dim+0.5,1) for dim in self._resolution], indexing="ij")
        return math.expand_dims(math.stack(idx_zyx, axis=-1), 0)

    def staggered_points(self, dimension):
        idx_zyx = np.meshgrid(*[np.arange(0.5,dim+1.5,1) if dim != dimension else np.arange(0,dim+1,1) for dim in self._resolution], indexing="ij")
        return math.expand_dims(math.stack(idx_zyx, axis=-1), 0)


    def indices(self):
        """
    Constructs a grid containing the index-location as components.
    Each index denotes the location within the tensor starting from zero.
    Indices are encoded as vectors in the index tensor.
        :param dtype: a numpy data type (default float32)
        :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
        """
        idx_zyx = np.meshgrid(*[range(dim) for dim in self._resolution], indexing="ij")
        return math.expand_dims(np.stack(idx_zyx, axis=-1))

    def shape(self, components=1, batch_size=1):
        return tensor_shape(batch_size, self._resolution, components)

    def staggered_shape(self, batch_size=1):
        return math.StaggeredGrid(tensor_shape(batch_size, self._resolution + 1, self.rank))
        return math.StaggeredGrid(tensor_shape(batch_size, self.dimensions + 1, self.rank))

    @staticmethod
    def equal(grid1, grid2):
        assert isinstance(grid1, Grid), 'Not a grid: %s' % type(grid1)
        assert isinstance(grid2, Grid), 'Not a grid: %s' % type(grid2)
        return grid1._dimensions == grid2._dimensions and grid1._box == grid2._box


def tensor_shape(batch_size, resolution, components):
    return np.concatenate([[batch_size], resolution, [components]])