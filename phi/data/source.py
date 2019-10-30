class UnknownShapeError(RuntimeError):

    def __init__(self, *args, **kwargs):
        RuntimeError.__init__(*args, ** kwargs)


class DataSource(object):

    def get(self, fieldname, frames):
        """
        Returns a NumPy array (or list of arrays) holding the data for a given fieldname.
        The batch size of the array (or list length) is equal to len(frames).

        :param fieldname: stream identifier string
        :param frames: list or tuple of frames
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

    def frames(self):
        """
        Returns an iterable object to read all frames of this DataSource.
        The iterator need not provide a length.
        """
        raise NotImplementedError(self)

    def shape(self, fieldname):
        """
        Returns a 1D tensor holding the tensor shape of a single data point from the fieldname.
        The first entry, specifying the batch dimension, must be equal to 1.

        If the shape cannot be determined, this method returns None.

        :param fieldname: stream identifier string
        """
        raise NotImplementedError(self)


class SceneSource(DataSource):

    def __init__(self, scene, frames=None, shape_map=None):
        self.scene = scene
        self._frames = frames
        self._shape_map = shape_map if shape_map is not None else dict()

    def get(self, fieldname, frames):
        for frame in frames:
            yield self.scene.read_array(fieldname, frame)

    def list_fieldnames(self):
        return self.scene.fieldnames

    def size(self, lookup=False):
        if self._frames is None and lookup:
            self._frames = self.scene.frames
        return len(self._frames) if self._frames is not None else None

    def frames(self):
        if self._frames is None:
            self._frames = self.scene.frames
        return self._frames

    def shape(self, fieldname):
        if fieldname in self._shape_map:
            return self._shape_map[fieldname]
        first_frame = next(self.frames())
        first_array = self.get(fieldname, [first_frame])
        shape = first_array.shape
        self._shape_map[fieldname] = shape
        return shape

    def __repr__(self):
        return "SceneSource[%s, frames=%s]" % (self.scene, self._frames)
