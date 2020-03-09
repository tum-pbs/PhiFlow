import warnings

import numpy as np
import tensorflow as tf

from phi import struct, math
from phi.data.fluidformat import _transform_for_writing, _writing_staticshape, read_zipped_array
from phi.math import is_static_shape
from phi.physics.world import StateProxy
from phi.struct.context import _unsafe

from .util import placeholder, dataset_handle


def build_graph_input(obj, input_type='placeholder', frames=None):
    """
Create placeholders for tensors in the supplied state.
    :param obj: struct or StateProxy
    :param input_type: 'placeholder' or 'dataset_handle'
    :param frames: Number of input frames. If not None, returns a list of input structs.
    :return:
      1. Valid state containing or derived from created placeholders or dataset handle
      2. dict mapping from placeholders to their default names (using struct.names)
    """
    if isinstance(obj, StateProxy):
        obj = obj.state
    assert struct.isstruct(obj)
    # --- Shapes and names ---
    writable_obj = _transform_for_writing(obj)
    shape = _writing_staticshape(obj)
    names = struct.names(writable_obj)
    if input_type == 'placeholder':
        if frames is not None: raise NotImplementedError()
        with _unsafe():
            placeholders = placeholder(shape)
        graph_in = struct.map(lambda x: x, placeholders)  # validates fields, splits staggered tensors
        return graph_in, {placeholders: names}
    elif input_type == 'dataset_handle':
        with _unsafe():
            dtypes = struct.dtype(writable_obj)
            dataset_nodes, iterator_handle = dataset_handle(shape, dtypes, frames=frames)
        graph_in = struct.map(lambda x: x, dataset_nodes)  # validates fields, splits staggered tensors
        shapes = struct.flatten(struct.staticshape(dataset_nodes), leaf_condition=is_static_shape)
        dtypes = struct.flatten(struct.dtype(dataset_nodes))
        return graph_in, {'names': struct.flatten(names), 'iterator_handle': iterator_handle, 'shapes': shapes, 'dtypes': dtypes, 'frames': frames}
    else:
        raise ValueError(input_type)


def load_state(state):
    warnings.warn("load_state() is deprecated, use build_graph_input() instead.")
    return build_graph_input(state)


def create_dataset(scene_sources, names, shapes, dtypes, batch_size, shuffle=False, frames=None, inner_frame_stride=1, outer_frame_stride=1, prefetch=2):
    concat_dataset = None
    count = 0
    for source in scene_sources:
        scene = source.scene
        nested_file_list = list(scene.data_paths(source.frames(), field_names=names))
        count += _example_count(len(nested_file_list), frames, inner_frame_stride, outer_frame_stride)
        scene_dataset = tf.data.Dataset.from_tensor_slices(nested_file_list)
        scene_dataset = scene_dataset.map(lambda *items: tuple(tf.py_func(_read_npy_files, items, dtypes)))
        if frames is not None:
            scene_dataset = stacked_window(scene_dataset, frames, outer_stride=outer_frame_stride, inner_stride=inner_frame_stride)
        if concat_dataset is None:
            concat_dataset = scene_dataset
        else:
            concat_dataset = tf.data.Dataset.concatenate(concat_dataset, scene_dataset)
    if shuffle:
        concat_dataset = concat_dataset.shuffle(count)
    dataset = concat_dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch)
    return dataset


def _read_npy_files(items):
    data = [read_zipped_array(item.decode())[0,...] for item in items]
    data = [math.to_float(array) for array in data]
    return data


def stacked_window(dataset, size, outer_stride=1, inner_stride=1):
    """
Combines lists of input elements into windows by adding a window dimension to the dataset.
All windows have the same number of elements.
    :param dataset: TensorFlow Dataset
    :param size: number of elements in each window
    :param outer_stride: skip windows; makes the outer dataset smaller
    :param inner_stride: element-stride inside each window
    :return: Dataset
    """
    dataset = dataset.window(size, shift=outer_stride, stride=inner_stride, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x)  # convert VariantDataset (window) to batch dimension
    dataset = dataset.batch(size)
    return dataset


def _example_count(length, frames, inner_stride, outer_stride):
    if frames is None:
        return length
    return (frames - (frames * inner_stride - 1)) // outer_stride