import warnings

import numpy as np
import os
import inspect

import webglviewer
from phi.math import GLOBAL_AXIS_ORDER as physics_config
from phi.field import CenteredGrid, StaggeredGrid


def load_sky(file, image_format=None, flatten=True, resolution=None, scale=1.0):
    # type: (str, str, bool, int, float) -> tuple
    assert image_format in ('cubemap', 'equiangular', None)
    import imageio
    imageio.plugins.freeimage.download()
    if not os.path.isabs(file):
        file = os.path.join(os.path.dirname(inspect.getfile(webglviewer)), file)
    image = imageio.imread(file)
    raw = np.array(image) * scale
    raw = np.concatenate((raw, np.ones((raw.shape[0], raw.shape[1], 1))), axis=-1)  # add alpha channel
    if image_format is None:
        image_format = detect_skybox_format(raw)
    if image_format == 'cubemap':
        images = split_cubemap(raw, resolution)
    else:
        images = equiangular_to_cubemap(raw, resolution)
    if flatten:
        images = [im.flatten() for im in images]
    return tuple(images)


def detect_skybox_format(raw):
    h, w = raw.shape[:2]
    if w == 2 * h:
        return 'equiangular'
    if h // 3 == w // 4:
        return 'cubemap'
    raise ValueError('Unknown skybox format. Shape=%s' % raw.shape)


def split_cubemap(raw, target_resolution):
    h, w = raw.shape[0] // 3, raw.shape[1] // 4
    assert w == h
    right = raw[h:2*h, 2*w:3*w]
    left = raw[h:2*h, :w]
    top = raw[:h, w:2*w]
    bottom = raw[2*h:, w:2*w]
    front = raw[h:2*h, w:2*w]
    back = raw[h:2*h, 3*w:]
    return right, left, top, bottom, front, back


def equiangular_to_cubemap(equiangular, resolution=None):
    shape = equiangular.shape
    h, w = shape[:2]

    if resolution is None:
        resolution = h // 2

    def lookup(theta, phi):
        """
        

        Args:
          theta: pi, pi]
          phi: 0, 2 pi]

        Returns:

        """
        flip = np.where((theta > np.pi/2) | (theta < -np.pi/2), -1, 1)
        phi *= flip
        theta = (theta * flip + np.pi/2) % np.pi - np.pi/2
        y = np.rint((-theta / np.pi + 0.5) * h).astype(np.int)
        y = np.clip(y, 0, h-1)
        x = np.rint((phi / 2 / np.pi + 0.5) * w).astype(np.int)
        x = x % w
        return equiangular[y, x, :]

    def face(theta0, phi0):
        x, y = np.meshgrid(*[np.linspace(-1, 1, resolution)] * 2)
        theta = np.arctan(y) + theta0
        phi = np.arctan(x) + phi0
        return theta, phi

    right = lookup(*face(0, np.pi/2))
    left = lookup(*face(0, -np.pi/2))
    top = lookup(*face(np.pi/2, 0))
    bottom = lookup(*face(-np.pi/2, 0))
    front = lookup(*face(0, 0))
    back = lookup(*face(0, np.pi))
    images = right, left, top, bottom, front, back
    return images


def default_sky():
    right = [.3, .5, .4]
    left = [.3, .4, .5]
    top = [.8, .8, 1]
    bottom = [.2, .15, .15]
    front = [.4, .5, .4]
    back = [.4, .4, .4]
    images = right, left, top, bottom, front, back
    images = [np.array(image + [1.0]) * 255 for image in images]
    return images


EMPTY_GRID = np.zeros([2, 2, 2], np.float32)


def webgl_prepare_data(field, config: dict, selection: dict) -> np.ndarray:
    if field is None:
        return EMPTY_GRID

    component = settings.get('component', 'abs')
    batch = settings.get('batch', 0)

    data = None

    if isinstance(field, CenteredGrid):
        if field.spatial_rank == 1:
            return EMPTY_GRID
        else:
            data = field.values

    elif isinstance(field, StaggeredGrid):
        if field.spatial_rank == 1:
            return EMPTY_GRID
        if component == 'vec2' or component == 'abs':
            data = field.at_centers().values
            component = 'abs'
        else:
            data = field.unstack()[{'z': physics_config.z, 'y': physics_config.y, 'x': physics_config.x}[component]].data

    if data is None:
        return EMPTY_GRID
    if not isinstance(data, np.ndarray):
        warnings.warn("WebGL plotting failed. Expected NumPy array but found '%s'" % data)
        return EMPTY_GRID

    if data.ndim == 4:
        data = np.stack([data] * 2, axis=1 if not physics_config.is_x_first else -2)

    data = data[min(batch, data.shape[0] - 1), ...]
    data = reduce_component(data, component)
    if not physics_config.is_x_first:
        data = np.transpose(data, axes=(1, 0, 2))
    else:
        data = np.transpose(data, axes=(1, 2, 0))
    data = np.abs(data) * 2
    return data
