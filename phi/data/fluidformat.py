# coding=utf-8
import inspect
import json
import logging
import os
import os.path
import re
import shutil
import sys
import warnings

import six
import numpy as np
from os.path import join, isfile, isdir

from phi import struct, math, __version__ as phi_version
from phi.physics import field
from phi.geom import GLOBAL_AXIS_ORDER as physics_config


def read_zipped_array(filename):
    file = np.load(filename)
    array = file[file.files[-1]]  # last entry in npz file has to be data array
    if array.shape[0] != 1 or len(array.shape) == 1:
        array = np.expand_dims(array, axis=0)
    if not physics_config.is_x_first and array.shape[-1] != 1:
        array = array[..., ::-1]  # component order in stored files is always XYZ
    return array


def write_zipped_array(filename, array):
    if array.shape[0] == 1 and len(array.shape) > 1:
        array = array[0, ...]
    if not physics_config.is_x_first and array.shape[-1] != 1:
        array = array[..., ::-1]  # component order in stored files is always XYZ
    np.savez_compressed(filename, array)


def _check_same_dimensions(arrays):
    for array in arrays:
        if array.shape[1:-1] != arrays[0].shape[1:-1]:
            raise ValueError("All arrays should have the same spatial dimensions, but got %s and %s" % (array.shape, arrays[0].shape))


def read_sim_frame(simpath, fieldnames, frame, set_missing_to_none=True):
    if isinstance(fieldnames, six.string_types):
        fieldnames = [fieldnames]
    for fieldname in fieldnames:
        filename = _filename(simpath, fieldname, frame)
        if os.path.isfile(filename):
            yield read_zipped_array(filename)
        else:
            if set_missing_to_none:
                yield None
            else:
                raise IOError("Missing data at frame %d: %s" % (frame, filename))


def write_sim_frame(simpath, arrays, fieldnames, frame, check_same_dimensions=False):
    if check_same_dimensions:
        _check_same_dimensions(arrays)
    os.path.isdir(simpath) or os.mkdir(simpath)
    if not isinstance(fieldnames, (tuple, list)) and not isinstance(arrays, (tuple, list)):
        fieldnames = [fieldnames]
        arrays = [arrays]
    filenames = [_filename(simpath, name, frame) for name in fieldnames]
    for i in range(len(arrays)):
        write_zipped_array(filenames[i], arrays[i])
    return filenames


def _filename(simpath, name, frame):
    return join(simpath, "%s_%06i.npz" % (name, frame))


def read_sim_frames(simpath, fieldnames=None, frames=None):
    if fieldnames is None:
        fieldnames = get_fieldnames(simpath)
    if not fieldnames:
        return []
    if frames is None:
        frames = get_frames(simpath, fieldnames[0])
    if isinstance(frames, int):
        frames = [frames]
    single_fieldname = isinstance(fieldnames, six.string_types)
    if single_fieldname:
        fieldnames = [fieldnames]

    field_lists = [[] for f in fieldnames]
    for i in frames:
        fields = list(read_sim_frame(simpath, fieldnames, i, set_missing_to_none=False))
        for j in range(len(fieldnames)):
            field_lists[j].append(fields[j])
    result = [np.concatenate(list, 0) for list in field_lists]
    return result if not single_fieldname else result[0]


def get_fieldnames(simpath):
    fieldnames_set = {f[:-11] for f in os.listdir(simpath) if f.endswith(".npz")}
    return sorted(fieldnames_set)


def first_frame(simpath, fieldname=None):
    return min(get_frames(simpath, fieldname))


def get_frames(simpath, fieldname=None, mode="intersect"):
    if fieldname is not None:
        all_frames = {int(f[-10:-4]) for f in os.listdir(simpath) if f.startswith(fieldname) and f.endswith(".npz")}
        return sorted(all_frames)
    else:
        frames_lists = [get_frames(simpath, fieldname) for fieldname in get_fieldnames(simpath)]
        if mode.lower() == "intersect":
            intersection = set(frames_lists[0]).intersection(*frames_lists[1:])
            return sorted(intersection)
        elif mode.lower() == "union":
            if not frames_lists:
                return []
            union = set(frames_lists[0]).union(*frames_lists[1:])
            return sorted(union)


def _copy_file(source, target):
    shutil.copy(source, target)
    try:
        shutil.copystat(source, target)
    except:
        warnings.warn('Could not copy file metadata to %s' % target)


class Scene(object):

    def __init__(self, dir, category, index):
        self.dir = dir
        self.category = category
        self.index = index
        self._properties = None

    @property
    def path(self):
        return join(self.dir, self.category, "sim_%06d" % self.index)

    def subpath(self, name, create=False):
        path = join(self.path, name)
        if create and not os.path.isdir(path):
            os.mkdir(path)
        return path

    def _init_properties(self):
        if self._properties is not None:
            return
        dfile = join(self.path, "description.json")
        if isfile(dfile):
            self._properties = json.load(dfile)
        else:
            self._properties = {}

    def exists_config(self):
        return isfile(join(self.path, "description.json"))

    @property
    def properties(self):
        self._init_properties()
        return self._properties

    @properties.setter
    def properties(self, dict):
        self._properties = dict
        with open(join(self.path, "description.json"), "w") as out:
            json.dump(self._properties, out, indent=2)

    def put_property(self, key, value):
        self._init_properties()
        self._properties[key] = value
        with open(join(self.path, "description.json"), "w") as out:
            json.dump(self._properties, out, indent=2)

    def read_sim_frames(self, fieldnames=None, frames=None):
        return read_sim_frames(self.path, fieldnames=fieldnames, frames=frames)

    def read_array(self, fieldname, frame):
        return next(read_sim_frame(self.path, [fieldname], frame, set_missing_to_none=False))

    def write_sim_frame(self, arrays, fieldnames, frame, check_same_dimensions=False):
        write_sim_frame(self.path, arrays, fieldnames, frame, check_same_dimensions=check_same_dimensions)

    def write(self, obj, names=None, frame=0):
        if struct.isstruct(obj):
            obj = _transform_for_writing(obj)
            if names is None:
                names = struct.names(obj)
            values = struct.flatten(obj)
            names = struct.flatten(names)
            names = [_slugify_filename(name) for name in names]
            self.write_sim_frame(values, names, frame)
        else:
            name = str(names) if names is not None else 'unnamed'
            self.write_sim_frame([obj], [name], frame)

    def read(self, obj, frame=0):
        if struct.isstruct(obj):
            obj = _transform_for_writing(obj)
            names = struct.flatten(obj)
            if not np.all([isinstance(n, six.string_types) for n in names]):
                names = struct.names(obj)
            data = struct.map(lambda name: self.read_array(_slugify_filename(name), frame), names)
            return data
        else:
            return self.read_array('unnamed', frame)

    @property
    def fieldnames(self):
        return get_fieldnames(self.path)

    @property
    def frames(self):
        return get_frames(self.path)

    def get_frames(self, mode="intersect"):
        return get_frames(self.path, None, mode)

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path

    def copy_calling_script(self, full_trace=False, include_context_information=True):
        script_paths = [frame[1] for frame in inspect.stack()]
        script_paths = list(filter(lambda path: not _is_phi_file(path), script_paths))
        script_paths = set(script_paths) if full_trace else [script_paths[0]]
        for script_path in script_paths:
            _copy_file(script_path, join(self.subpath('src', create=True), os.path.basename(script_path)))
        if include_context_information:
            with open(join(self.subpath('src', create=True), 'context.json'), 'w') as context_file:
                json.dump({
                    'phi_version': phi_version,
                    'argv': sys.argv
                }, context_file)

    def copy_src(self, path, only_external=True):
        if not only_external or not _is_phi_file(path):
            _copy_file(path, join(self.subpath('src', create=True), os.path.basename(path)))

    def mkdir(self, subdir=None):
        path = self.path
        isdir(path) or os.mkdir(path)
        if subdir is not None:
            subpath = join(path, subdir)
            isdir(subpath) or os.mkdir(subpath)

    def remove(self):
        if isdir(self.path):
            shutil.rmtree(self.path)

    def data_paths(self, frames, field_names):
        for frame in frames:
            yield tuple([_filename(self.path, name, frame) for name in field_names])

    @staticmethod
    def create(directory, category=None, count=1, mkdir=True, copy_calling_script=True):
        if count > 1:
            scenes = []
            for _ in range(count):
                scenes.append(Scene.create(directory, category, 1, mkdir, copy_calling_script))
            return SceneBatch(scenes)
        # Single scene
        directory = os.path.expanduser(directory)
        if category is None:
            category = os.path.basename(directory)
            directory = os.path.dirname(directory)
        else:
            category = slugify(category)

        scenedir = join(directory, category)
        if not isdir(scenedir):
            os.makedirs(scenedir)
            next_index = 0
        else:
            indices = [int(name[4:]) for name in os.listdir(scenedir) if name.startswith("sim_")]
            if not indices:
                next_index = 0
            else:
                next_index = max(indices) + 1
        scene = Scene(directory, category, next_index)
        if mkdir:
            scene.mkdir()
        if copy_calling_script:
            try:
                assert mkdir
                scene.copy_calling_script()
            except IOError as err:
                warnings.warn('Failed to copy calling script to scene during Scene.create().\nCause: %s' % err)
        return scene

    @staticmethod
    def list(directory, category=None, indexfilter=None, max_count=None):
        directory = os.path.expanduser(directory)
        if not category:
            root_path = directory
            category = os.path.basename(directory)
            directory = os.path.dirname(directory)
        else:
            root_path = join(directory, category)
        if not os.path.isdir(root_path):
            return []
        indices = [int(sim[4:]) for sim in os.listdir(root_path) if sim.startswith("sim_")]
        if indexfilter:
            indices = [i for i in indices if indexfilter(i)]
        if max_count and len(indices) >= max_count:
            indices = indices[0:max_count]
        indices = sorted(indices)
        if len(indices) == 0:
            logging.warning("No simulations sim_XXXXXX found in '%s'" % root_path)
        return [Scene(directory, category, scene_index) for scene_index in indices]

    @staticmethod
    def at(sim_dir):
        sim_dir = os.path.expanduser(sim_dir)
        if sim_dir[-1] == '/':  # remove trailing backslash
            sim_dir = sim_dir[0:-1]
        dirname = os.path.basename(sim_dir)
        if not dirname.startswith("sim_"):
            raise ValueError("%s with dir %s is not a valid scene directory." % (sim_dir,dirname))
        category_directory = os.path.dirname(sim_dir)
        category = os.path.basename(category_directory)
        directory = os.path.dirname(category_directory)
        index = int(dirname[4:])
        return Scene(directory, category, index)


class SceneBatch(Scene):

    def __init__(self, scenes):
        Scene.__init__(self, scenes[0].dir, scenes[0].category, scenes[0].index)
        self.scenes = scenes

    @property
    def batch_size(self):
        return len(self.scenes)

    def write_sim_frame(self, arrays, fieldnames, frame, check_same_dimensions=False):
        for array in arrays:
            assert array.shape[0] == self.batch_size or array.shape[0] == 1,\
                'Wrong batch size: %d but %d scenes' % (array.shape[0], self.batch_size)
        for i, scene in enumerate(self.scenes):
            array_slices = [(array[i, ...] if array.shape[0] > 1 else array[0, ...]) for array in arrays]
            scene.write_sim_frame(array_slices, fieldnames, frame=frame, check_same_dimensions=check_same_dimensions)

    def read_sim_frames(self, fieldnames=None, frames=None):
        raise NotImplementedError()

    def read_array(self, fieldname, frame):
        return np.concatenate([scene.read_array(fieldname, frame) for scene in self.scenes])


def _transform_for_writing(obj):
    def f(value):
        if isinstance(value, field.StaggeredGrid):
            return value.staggered_tensor()
        if isinstance(value, field.CenteredGrid):
            return value.data
        else:
            return value
    data = struct.map(f, obj, lambda x: isinstance(x, (field.StaggeredGrid, field.CenteredGrid)), content_type='format')
    return data


def _writing_staticshape(obj):
    def f(value):
        if isinstance(value, field.StaggeredGrid):
            shape = math.staticshape(value.staggered_tensor())
            return (value._batch_size,) + shape[1:]
        if isinstance(value, field.CenteredGrid):
            return value.staticshape.data
        else:
            return math.staticshape(value)
    data = struct.map(f, obj, lambda x: isinstance(x, (field.StaggeredGrid, field.CenteredGrid)), content_type='format_staticshape')
    return data


def _slugify_filename(struct_name):
    struct_name = struct_name.replace('._', '.').replace('.', '_')
    if struct_name.startswith('_'):
        struct_name = struct_name[1:]
    return struct_name


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    for greek_letter, name in greek.items():
        value = value.replace(greek_letter, name)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


greek = {
    u'Α': 'Alpha',      u'α': 'alpha',
    u'Β': 'Beta',       u'β': 'beta',
    u'Γ': 'Gamma',      u'γ': 'gamma',
    u'Δ': 'Delta',      u'δ': 'delta',
    u'Ε': 'Epsilon',    u'ε': 'epsilon',
    u'Ζ': 'Zeta',       u'ζ': 'zeta',
    u'Η': 'Eta',        u'η': 'eta',
    u'Θ': 'Theta',      u'θ': 'theta',
    u'Ι': 'Iota',       u'ι': 'iota',
    u'Κ': 'Kappa',      u'κ': 'kappa',
    u'Λ': 'Lambda',     u'λ': 'lambda',
    u'Μ': 'Mu',         u'μ': 'mu',
    u'Ν': 'Nu',         u'ν': 'nu',
    u'Ξ': 'Xi',         u'ξ': 'xi',
    u'Ο': 'Omicron',    u'ο': 'omicron',
    u'Π': 'Pi',         u'π': 'pi',
    u'Ρ': 'Rho',        u'ρ': 'rho',
    u'Σ': 'Sigma',      u'σ': 'sigma',
    u'Τ': 'Tau',        u'τ': 'tau',
    u'Υ': 'Upsilon',    u'υ': 'upsilon',
    u'Φ': 'Phi',        u'φ': 'phi',
    u'Χ': 'Chi',        u'χ': 'chi',
    u'Ψ': 'Psi',        u'ψ': 'psi',
    u'Ω': 'Omega',      u'ω': 'omega',
}


def _is_phi_file(path):
    path, name = os.path.split(path)
    if name == 'phi':
        return True
    elif path == '' or name == '':
        return False
    else:
        return _is_phi_file(path)
