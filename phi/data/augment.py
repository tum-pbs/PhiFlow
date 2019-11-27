from .stream import DerivedStream


class Augmentstream(DerivedStream):

    def __init__(self, aug_dimensions, field, affect_flags=(DATAFLAG_TRAIN,)):
        DerivedStream.__init__(self, [field])
        self.field = self.inputs[0]
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

    def size(self, datasource, lookup=False):
        return self.field.size(datasource, lookup=lookup) * (self.augmentation_factor if self.affects(datasource) else 1)

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


class AxisFlip(Augmentstream):

    def __init__(self, flip_dimensions, field, flip_vectors=True, affect_flags=(DATAFLAG_TRAIN,)):
        Augmentstream.__init__(self, flip_dimensions, field, affect_flags)
        self.flip_vectors = flip_vectors

    def augment_single(self, array, perm):
        slices = [slice(None, None, -1) if d >= 1 and perm & 2 ** (d - 1) != 0 else slice(None) for d in range(len(array.shape))]
        array = array[slices]
        if self.flip_vectors and array.shape[-1] == len(array.shape) - 2:
            flipped_components = [len(array.shape) - d - 3 for d in self.aug_dimensions if perm & 2 ** (d) != 0]
            array[..., flipped_components] *= -1
        return array
