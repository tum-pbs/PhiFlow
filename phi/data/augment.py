from phi.data import *


class AugmentChannel(DerivedChannel):

    def __init__(self, aug_dimensions, field, affect_flags=(DATAFLAG_TRAIN,)):
        DerivedChannel.__init__(self, [field])
        self.field = self.input_fields[0]
        self.affect_flags = affect_flags
        self.aug_dimensions = range(aug_dimensions) if isinstance(aug_dimensions, int) else aug_dimensions
        self.augmentation_factor = 2 ** len(self.aug_dimensions)

    def affects(self, datasource):
        if not self.affect_flags:
            return True
        for flag in self.affect_flags:
            if flag in datasource.flags:
                return True
        return False

    def size(self, datasource):
        return self.field.size(datasource) * (self.augmentation_factor if self.affects(datasource) else 1)

    def shape(self, datasource):
        return self.field.shape(datasource)

    def get(self, datasource, indices):
        if not self.affects(datasource):
            return self.field.get(datasource, indices)
        else:
            return self._augment_interleave(datasource, indices)

    def _augment_interleave(self, datasource, indices):
        arrays = self.field.get(datasource, [i // self.augmentation_factor for i in indices])
        for array, index in zip(arrays, indices):
            perm = index % self.augmentation_factor
            if perm != 0:
                array = self.augment_single(array, perm)
            yield array

    def augment_single(self, array, perm):
        raise NotImplementedError()


class AxisFlip(AugmentChannel):

    def __init__(self, flip_dimensions, field, flip_vectors=True, affect_flags=(DATAFLAG_TRAIN,)):
        AugmentChannel.__init__(self, flip_dimensions, field, affect_flags)
        self.flip_vectors = flip_vectors

    def augment_single(self, array, perm):
        slices = [slice(None, None, -1) if d >= 1 and perm & 2 ** (d - 1) != 0 else slice(None) for d in range(len(array.shape))]
        array = array[slices]
        if self.flip_vectors and array.shape[-1] == len(array.shape) - 2:
            flipped_components = [len(array.shape) - d - 3 for d in self.aug_dimensions if perm & 2 ** (d) != 0]
            array[..., flipped_components] *= -1
        return array


# class SpatialShift(AugmentChannel):
#
#     def __init__(self, shift_dimensions, shift, field, padding="symmetric", affect_flags=(DATAFLAG_TRAIN,)):
#         AugmentChannel.__init__(self, shift_dimensions, field, affect_flags)
#         self.shift = shift
#         self.padding = padding
#
#     def shape(self, datasource):
#         input_shape = self.field.shape(datasource)
#         if self.padding is not None:
#             return input_shape
#         else:
#             input_shape = np.array(input_shape)
#             input_shape[1:-1] -= 2
#             return input_shape
#
#     def augment_single(self, array, perm):
#         slices = [slice(None, None, -1) if d >= 1 and perm & 2 ** (d - 1) != 0 else slice(None) for d in range(len(array.shape))]
#
#     def velocity_adjustment(self, field):