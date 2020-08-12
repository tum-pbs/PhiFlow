import os
import numpy as np

from phi.physics.field import CenteredGrid, StaggeredGrid
from phi.physics.field.staggered_grid import stack_staggered_components


# Views
FRONT = 'front'
RIGHT = 'right'
TOP = 'top'

# Vector display
LENGTH = 'length'
VECTOR2 = 'vec2'


class PlotlyFigureBuilder(object):

    def __init__(self,
                 batches=slice(None),
                 depths=slice(None),
                 antisymmetry=False,
                 view=FRONT,
                 component=LENGTH,
                 draw_arrows_backward=True,
                 max_vector_resolution=18,
                 max_resolution=128):
        self.batches = batches
        self.depths = depths
        self.antisymmetry = antisymmetry
        self.view = view
        self.component = component
        self.draw_arrows_backward = draw_arrows_backward
        self.max_vector_resolution = max_vector_resolution
        self.max_resolution = max_resolution

    def select_batch(self, batch):
        if batch is None:
            self.batches = slice(None)
        elif isinstance(batch, int):
            self.batches = [batch]
        else:
            self.batches = batch

    def select_depth(self, depth):
        if depth is None:
            self.depths = slice(None)
        elif isinstance(depth, int):
            self.depths = [depth]
        else:
            self.depths = depth

    def save_figures(self, directory, fieldname, time, data, same_scale_data=None):
        import matplotlib.pyplot as plt
        batches = self.batches if self.batches is not None else range(data.shape[0])
        for batch in batches:
            if len(data.shape) == 5:
                for depth in self.get_selected_slices(data.shape):
                    path = os.path.join(directory, '%s_batch%04d_depth%04d_%04d.png' % (fieldname, batch, depth, time))
                    fig = self.create_figure(data, batch=batch, depth=depth, same_scale_data=same_scale_data, library='matplotlib')
                    plt.savefig(path)
                    plt.close()
                    yield path
            else:
                path = os.path.join(directory, '%s_batch%04d_%04d.png' % (fieldname, batch, time))
                fig = self.create_figure(data, batch=batch, same_scale_data=same_scale_data, library='matplotlib')
                plt.savefig(path)
                plt.close()
                yield path

    def get_selected_slices(self, shape):
        try:
            selected_depths = np.arange(self.slice_count(shape))[self.depths]
        except:
            selected_depths = [self.slice_count(shape) - 1]
        return selected_depths

    def slice_dims(self, data):
        if isinstance(data, CenteredGrid):
            return data.data.shape
        if isinstance(data, StaggeredGrid):
            return data.data[0].data.shape
        else:
            return data.shape

    def create_figure(self, data, same_scale_data=None, batch=None, depth=None, library='matplotlib'):
        shape = self.slice_dims(data)
        # Determine batch
        if shape[0] == 1:
            batch = 0
        if batch is None:
            try:
                selected_batches = np.arange(shape[0])[self.batches]
                if len(selected_batches) != 1:
                    raise ValueError('no batch specified and default batches contains more than one element')
            except:
                return None
            batch = selected_batches[0]
        # Determine slice
        if depth is None and len(shape) == 5:
            selected_depths = self.get_selected_slices(shape)
            if len(selected_depths) != 1:
                raise ValueError('no depth specified and default depths contains more than one element')
            depth = selected_depths[0]

        # special handling for staggered grids and anti symmetry
        if isinstance(data, StaggeredGrid):
            if not self.antisymmetry:
                data = data.staggered_tensor()
            else:
                dims = len(data.data[0].data.shape) - 2  # any better way to get this?
                dataxyz = []
                for i in range(dims):
                    c = data.data[i].data
                    factor = -1. if i == (dims - 1) else 1.  # add (instead of subtract) for X dim
                    cdiff = c[..., ::-1,0:1] - (c[...,0:1] * factor)
                    dataxyz.append(cdiff)
                data = stack_staggered_components(dataxyz)
            shape = data.shape
        else:
            if isinstance(data, CenteredGrid):
                data = data.data
            if self.antisymmetry:
                data = data - data[..., ::-1, :]

        # Select batch
        if batch < shape[0]:
            data = data[batch, ...]
        else:
            return {'data': [{'type': 'heatmap', 'z': [[0]]}]}

        if np.issubdtype(data.dtype, np.complex):
            data = np.real(data)

        # 1D graph
        if len(shape) == 3:
            return self.graphs(data, library)

        # 3D projection / Select depth
        if len(shape) == 5:
            if self.view == FRONT:
                data = data[:, min(depth, shape[2]), :, :]
            elif self.view == RIGHT:
                data = data[:, :, min(depth, shape[3]), :]
                data = np.transpose(data, axes=(1, 0, 2))
            elif self.view == TOP:
                data = data[min(depth, shape[1]), :, :, :]
            else:
                data = data[0, ...]

        # Create figure
        component = 0 if shape[-1] == 1 else self.component

        if component == VECTOR2:
            # Downsample
            while np.prod(data.shape[:-1]) > self.max_vector_resolution ** 2:
                data = data[::2, ::2, :] * 0.5
            data = data[..., ::-1]
            return self.draw_vector_field(data, library)

        elif component == LENGTH:
            if shape[-1] == 3:
                data = np.sqrt(data[..., 0:1]**2 + data[..., 1:2]**2 + data[..., 2:3]**2)
            else:
                data = np.sqrt(data[..., 0:1]**2 + data[..., 1:2]**2)
        else:
            # Single vector component
            if component >= shape[-1]:
                data = np.zeros_like(data[..., 0:1])
            else:
                data = data[..., shape[-1] - 1 - component:shape[-1] - component]

        # Downsample
        while np.prod(data.shape[:-1]) > self.max_resolution ** 2:
            data = data[::2, ::2, :]
        if same_scale_data is not None:
            return self.heatmap(data[..., 0], library, minmax=global_minmax(same_scale_data))
        else:
            return self.heatmap(data[..., 0], library)

    def slice_count(self, shape):
        if len(shape) <= 4:
            return 1
        if self.view == FRONT:
            return shape[1]
        elif self.view == TOP:
            return shape[2]
        elif self.view == RIGHT:
            return shape[3]
        else:
            raise ValueError('Illegal view: %s' % self.view)

    def heatmap(self, z, library, minmax=None):
        if library == 'dash':
            args = {'z': z, 'type': 'heatmap'}
            if minmax is not None:
                args['zmin'] = minmax[0]
                args['zmax'] = minmax[1]
            return {'data': [args]}
        elif library == 'matplotlib':
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(z, cmap='bwr', origin='lower')
            return fig
        else:
            raise NotImplementedError()

    def graphs(self, data, library):
        x = np.arange(data.shape[0])
        if library == 'dash':
            graphs = [{'mode': 'markers+lines', 'type': 'scatter', 'x': x, 'y': data[:, i]} for i in range(data.shape[-1])]
            return {'data': graphs}
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            for i in range(data.shape[-1]):
                plt.plot(x, data[:,i])
            return fig

    def draw_vector_field(self, vector_field, library):
        import plotly.figure_factory as ff
        x, y = np.meshgrid(np.arange(0, vector_field.shape[1], 1), np.arange(0, vector_field.shape[0], 1))
        if library == 'dash':
            if self.draw_arrows_backward:
                return ff.create_quiver(x - vector_field[..., 0], y - vector_field[..., 1], vector_field[..., 0], vector_field[..., 1], scale=1.0)
            else:
                return ff.create_quiver(x, y, vector_field[..., 0], vector_field[..., 1], scale=1.0)
        else:
            raise NotImplementedError()

    def empty_figure(self, library):
        if library == 'dash':
            return {
                'data': [{'z': None, 'type': 'heatmap'}]
            }
        elif library == 'matplotlib':
            import matplotlib.pyplot as plt
            fig = plt.figure()
            return fig


def global_minmax(arrays):
    global_min = np.minimum(*[np.min(data) for data in arrays])
    global_max = np.maximum(*[np.max(data) for data in arrays])
    return global_min, global_max
