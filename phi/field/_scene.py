# coding=utf-8
import inspect
import json
import logging
import os
import re
import shutil
import sys
import warnings
from os.path import join, isfile, isdir, abspath, expanduser, basename, dirname, split

import numpy as np

from phi import struct, math, __version__ as phi_version
from ._field import Field, SampledField
from ._field_io import read, write


def read_sim_frame(directory: math.Tensor,
                   names: str or tuple or list or dict or struct.Struct,
                   frame: int,
                   convert_to_backend=True):
    def single_read(name):
        name = _slugify_filename(name)
        files = math.map(lambda dir_: _filename(dir_, name, frame), directory)
        return read(files, convert_to_backend=convert_to_backend)

    return struct.map(single_read, names)


def write_sim_frame(directory: math.Tensor,
                    fields: Field or tuple or list or dict or struct.Struct,
                    frame: int,
                    names: str or tuple or list or struct.Struct or None = None):
    """
    Write a Field or structure of Fields to files.
    The filenames are created from the provided names and the frame index in accordance with the
    scene format specification at https://tum-pbs.github.io/PhiFlow/Scene_Format_Specification.html .

    This method can be used in batch mode.
    Batch mode is active if a list of directories is given instead of a single directory.
    Then, all fields are unstacked along the batch_dim dimension and matched with the directories list.

    Args:
        directory: directory name or list of directories.
            If a list is provided, all fields are unstacked along batch_dim and matched with their respective directory.
        fields: single field or structure of Fields to save.
        frame: Number < 1000000, typically time step index.
        names: (Optional) Structure matching fields, holding the filename for each respective Field.
            If not provided, names are automatically generated based on the structure of fields.
    """
    if names is None:
        names = struct.names(fields)
    if frame > 1000000:
        warnings.warn(f"frame too large: {frame}. Data will be saved but filename might cause trouble in the future.")

    def single_write(f, name):
        name = _slugify_filename(name)
        files = math.map(lambda dir_: _filename(dir_, name, frame), directory)
        if isinstance(f, SampledField):
            write(f, files)
        elif isinstance(f, math.Tensor):
            raise NotImplementedError()
        elif isinstance(f, Field):
            raise ValueError("write_sim_frame: only SampledField instances are saved. Resample other Fields before saving them.")
        else:
            raise ValueError(f"write_sim_frame: only SampledField instances can be saved but got {f}")

    struct.foreach(single_write, fields, names)


def _filename(simpath, name, frame):
    return join(simpath, "%s_%06i.npz" % (name, frame))


def get_fieldnames(simpath) -> tuple:
    fieldnames_set = {f[:-11] for f in os.listdir(simpath) if f.endswith(".npz")}
    return tuple(sorted(fieldnames_set))


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


class Scene(object):
    """
    Provides methods for reading and writing simulation data.

    See the format documentation at https://tum-pbs.github.io/PhiFlow/Scene_Format_Specification.html .

    All data of a `Scene` is located inside a single directory with name `sim_xxxxxx` where `xxxxxx` is the `id`.
    The data of the scene is organized into NumPy files by *name* and *frame*.

    To create a new scene, use `Scene.create()`.
    To reference an existing scene, use `Scene.at()`.
    To list all scenes within a directory, use `Scene.list()`.
    """

    def __init__(self, paths: str or math.Tensor):
        self._paths = math.tensor(paths)
        self._properties = None

    @property
    def shape(self):
        return self._paths.shape

    @property
    def is_batch(self):
        return self._paths.rank > 0

    @property
    def path(self) -> str:
        """
        Relative path of the scene directory.
        This property only exists for single scenes, not scene batches.
        """
        assert not self.is_batch, "Scene.path is not defined for scene batches."
        return self._paths.native()

    @property
    def paths(self) -> math.Tensor:
        return self._paths

    @staticmethod
    def batch_stack(*scenes: 'Scene', dim: str = 'batch') -> 'Scene':
        return Scene(math.batch_stack([s._paths for s in scenes], dim))

    @staticmethod
    def create(parent_directory: str,
               shape: math.Shape = math.EMPTY_SHAPE,
               copy_calling_script=True,
               **dimensions) -> 'Scene':
        """
        Creates a new `Scene` or a batch of new scenes inside `parent_directory`.

        See Also:
            `Scene.at()`, `Scene.list()`.

        Args:
            parent_directory: Directory to hold the new `Scene`. If it doesn't exist, it will be created.
            shape: Determines number of scenes to create. Multiple scenes will be represented by a `Scene` with `is_batch=True`.
            copy_calling_script: Whether to copy the Python file that invoked this method into the `src` folder of all created scenes.
                See `Scene.copy_calling_script()`.
            dimensions: Additional batch dimensions

        Returns:
            Single `Scene` object representing the new scene(s).
        """
        shape = (shape & math.shape(**dimensions)).to_batch()
        parent_directory = expanduser(parent_directory)
        abs_dir = abspath(parent_directory)
        if not isdir(abs_dir):
            os.makedirs(abs_dir)
            next_id = 0
        else:
            indices = [int(name[4:]) for name in os.listdir(abs_dir) if name.startswith("sim_")]
            next_id = max([-1] + indices) + 1
        ids = math.tensor(tuple(range(next_id, next_id + shape.volume))).vector.split(shape)
        paths = math.map(lambda id_: join(parent_directory, f"sim_{id_:06d}"), ids)
        scene = Scene(paths)
        scene.mkdir()
        if copy_calling_script:
            try:
                scene.copy_calling_script()
            except IOError as err:
                warnings.warn(f"Failed to copy calling script to scene during Scene.create(): {err}")
        return scene

    @staticmethod
    def list(parent_directory: str,
             include_other: bool = False,
             dim: str or None = None) -> 'Scene' or tuple:
        """
        Lists all scenes inside the given directory.

        See Also:
            `Scene.at()`, `Scene.create()`.

        Args:
            parent_directory: Directory that contains scene folders.
            include_other: Whether folders that do not match the scene format should also be treated as scenes.
            dim: Stack dimension. If None, returns tuple of `Scene` objects. Otherwise, returns a scene batch with this dimension.

        Returns:
            `tuple` of scenes.
        """
        parent_directory = expanduser(parent_directory)
        abs_dir = abspath(parent_directory)
        if not isdir(abs_dir):
            return ()
        names = [sim for sim in os.listdir(abs_dir) if sim.startswith("sim_") or (include_other and isdir(join(abs_dir, sim)))]
        if dim is None:
            return tuple(Scene(join(parent_directory, name)) for name in names)
        else:
            paths = math.tensor([join(parent_directory, name) for name in names], dim)
            return Scene(paths)

    @staticmethod
    def at(directory: str or math.Tensor, id: int or math.Tensor or None = None) -> 'Scene':
        """
        Creates a `Scene` for an existing directory.

        See Also:
            `Scene.create()`, `Scene.list()`.

        Args:
            directory: Either directory containing scene folder if `id` is given, or scene path if `id=None`.
            id: (Optional) Scene `id`, will be determined from `directory` if not specified.

        Returns:
            `Scene` object for existing scene.
        """
        directory = math.map(lambda d: expanduser(d), math.tensor(directory))
        if id is None:
            paths = directory
        else:
            id = math.tensor(id)
            paths = directory._op2(id, None, lambda d_, id_: join(directory, f"sim_{id:06d}"))
        # test all exist
        for path in math.flatten(paths):
            if not isdir(path):
                raise IOError(f"There is no scene at '{path}'")
        return Scene(paths)

    def subpath(self, name: str, create: bool = False) -> str or tuple:
        """
        Resolves the relative path `name` with this `Scene` as the root folder.

        Args:
            name: Relative path with this `Scene` as the root folder.
            create: Whether to create a directory of that name.

        Returns:
            Relative path including the path to this `Scene`.
            In batch mode, returns a `tuple`, else a `str`.
        """
        def single_subpath(path):
            path = join(path, name)
            if create and not isdir(path):
                os.mkdir(path)
            return path

        result = math.map(single_subpath, self._paths)
        if result.rank == 0:
            return result.native()
        else:
            return result

    def _init_properties(self):
        if self._properties is not None:
            return
        dfile = join(self.path, "description.json")
        if isfile(dfile):
            self._properties = json.load(dfile)
        else:
            self._properties = {}

    def exists_config(self):
        """ Tests if the configuration file *description.json* exists. In batch mode, tests if any configuration exists. """
        if isinstance(self.path, str):
            return isfile(join(self.path, "description.json"))
        else:
            return any(isfile(join(p, "description.json")) for p in self.path)

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

    def write_sim_frame(self, arrays, fieldnames, frame):
        write_sim_frame(self._paths, arrays, names=fieldnames, frame=frame)

    def write(self, data: dict, frame=0):
        """
        Writes fields to this scene.
        One NumPy file will be created for each `phi.field.Field`

        See Also:
            `Scene.read()`.

        Args:
            data: `dict` mapping field names to `Field` objects that can be written using `phi.field.write()`.
            frame: Frame number.
        """
        write_sim_frame(self._paths, data, names=None, frame=frame)

    def read_array(self, field_name, frame):
        return read_sim_frame(self._paths, field_name, frame=frame)

    # def read_sim_frames(self, fieldnames=None, frames=None):
    #     return read_sim_frames(self.path, fieldnames=fieldnames, frames=frames, batch_dim=self.batch_dim)

    def read(self, names: str or tuple or list, frame=0, convert_to_backend=True):
        """
        Reads one or multiple fields from disc.

        See Also:
            `Scene.write()`.

        Args:
            names: Single field name or sequence of field names.
            frame: Frame number.
            convert_to_backend: Whether to convert the read data to the data format of the default backend, e.g. TensorFlow tensors.

        Returns:
            Single `phi.field.Field` or sequence of fields, depending on the type of `names`.
        """
        return read_sim_frame(self._paths, names, frame=frame, convert_to_backend=convert_to_backend)

    @property
    def fieldnames(self) -> tuple:
        """ Determines all field names present in this `Scene`, independent of frame. """
        return get_fieldnames(self.path)

    @property
    def frames(self):
        """ Determines all frame numbers present in this `Scene`, independent of field names. See `Scene.get_frames()`. """
        return get_frames(self.path)

    def get_frames(self, mode="intersect"):
        return get_frames(self.path, None, mode)

    def __repr__(self):
        return repr(self.paths)

    def __eq__(self, other):
        return isinstance(other, Scene) and math.all(other._paths == self._paths)

    def copy_calling_script(self, full_trace=False, include_context_information=True):
        """
        Copies the Python file that called this method into the `src` folder of this `Scene`.

        In batch mode, the script is copied to all scenes.

        Args:
            full_trace: Whether to include scripts that indirectly called this method.
            include_context_information: If True, writes the phiflow version and `sys.argv` into `context.json`.
        """
        script_paths = [frame[1] for frame in inspect.stack()]
        script_paths = list(filter(lambda path: not _is_phi_file(path), script_paths))
        script_paths = set(script_paths) if full_trace else [script_paths[0]]
        for path in math.flatten(self._paths):
            self.subpath('src', create=True)
            for script_path in script_paths:
                shutil.copy(script_path, join(path, 'src', basename(script_path)))
            if include_context_information:
                with open(join(path, 'src', 'context.json'), 'w') as context_file:
                    json.dump({
                        'phi_version': phi_version,
                        'argv': sys.argv
                    }, context_file)

    def copy_src(self, path, only_external=True):
        if not only_external or not _is_phi_file(path):
            shutil.copy(path, join(self.subpath('src', create=True), basename(path)))

    def mkdir(self):
        for path in math.flatten(self._paths):
            isdir(path) or os.mkdir(path)

    def remove(self):
        """ Deletes the scene directory and all contained files. """
        for p in math.flatten(self._paths):
            p = abspath(p)
            if isdir(p):
                shutil.rmtree(p)

    def data_paths(self, frames, field_names):
        for frame in frames:
            yield tuple([_filename(self.path, name, frame) for name in field_names])


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
    path, name = split(path)
    if name == 'phi':
        return True
    elif path == '' or name == '':
        return False
    else:
        return _is_phi_file(path)
