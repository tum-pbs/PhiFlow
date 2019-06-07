from numpy import inf
from .fluidformat import Scene


class UnknownShapeError(RuntimeError):

    def __init__(self, *args, **kwargs):
        RuntimeError.__init__(*args, ** kwargs)


class DataSource(object):

    def get(self, fieldname, indices):
        """
Returns a NumPy array (or list of arrays) holding the data for a given fieldname.
The batch size of the array (or list length) is equal to len(indices).
        :param fieldname: field identifier string
        :param indices: list or tuple of indices
        """
        raise NotImplementedError(self)

    def list_fieldnames(self):
        """
Returns a list of all fieldnames contained in this source.
        """
        raise NotImplementedError(self)

    def size(self, lookup=False):
        """
Returns the number of data points in this source.

If the source is infinite, returns numpy.inf.
If the size is unknown at this point and lookup=False, returns None.
        :param lookup: If the size should be determined if unknown
        """
        raise NotImplementedError(self)
    
    def indices(self):
        """
Returns an iteratable object to read all indices of this DataSource.
The iterator need not provide a length.
        """
        raise NotImplementedError(self)

    def shape(self, fieldname):
        """
Returns a 1D tensor holding the tensor shape of a single data point from the fieldname.
The first entry, specifying the batch dimension, must be equal to 1.

If the shape cannot be determined, this method returns None.
        :param fieldname: field identifier string
        """
        raise NotImplementedError(self)




class SceneSource(DataSource):

    def __init__(self, scene, indices=None, shape=None):
        self.scene = scene
        self._indices = indices
        self._shape = shape

    def get(self, fieldname, indices):
        return self.scene.read_sim_frames([fieldname], indices)

    def list_fieldnames(self):
        return self.scene.fieldnames

    def size(self, lookup=False):
        if self._indices is None and lookup:
            self._indices = self.scene.indices
        return len(self._indices) if self._indices is not None else None

    def indices(self):
        return self._indices

    def shape(self, fieldname):
        if self._shape is not None:
            return self._shape
        if self.size(lookup=True) is None:
            return None
        first_index = next(self.indices())
        first_array = self.get(fieldname, [first_index])
        return first_array.shape

    def __repr__(self):
        return "SceneSource[%s, indices=%s, shape=%s]" % (self.scene, self._indices, self._shape)

    @staticmethod
    def list(directory, assume_same_indices=True, assume_same_shapes=True):
        raise NotImplementedError()
