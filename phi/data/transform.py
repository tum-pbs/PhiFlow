from phi.math.nd import downsample2x

from .stream import DerivedStream


class Downsample(DerivedStream):

    def frames(self, datasource):
        raise NotImplementedError()

    def __init__(self, field):
        DerivedStream.__init__(self, [field])
        self.field = self.inputs[0]

    def size(self, datasource, lookup=False):
        return self.field.size(datasource)

    def shape(self, datasource):
        in_shape = self.field.shape(datasource)
        return [in_shape[0]] + [s // 2 for s in in_shape[1:-1]] + [in_shape[-1]]

    def get(self, datasource, indices):
        data = self.field.get(datasource, indices)
        for array in data:
            yield downsample2x(array)
