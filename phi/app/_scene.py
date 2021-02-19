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

from phi import struct, math, __version__ as phi_version, field


def read_sim_frame(directory: str or tuple or list,
                   names: str or tuple or list or dict or struct.Struct,
                   frame: int,
                   batch_dim: str = 'batch',
                   convert_to_backend=True):
    assert isinstance(directory, (str, tuple, list))
    batch_mode = isinstance(directory, (tuple, list))

    def single_read(name):
        name = _slugify_filename(name)
        if batch_mode:
            file = [_filename(dir_, name, frame) for dir_ in directory]
            fields = [field.read(f, convert_to_backend=convert_to_backend) for f in file]
            return field.batch_stack(*fields, dim=batch_dim)
        else:
            file = _filename(directory, name, frame)
            return field.read(file, convert_to_backend=convert_to_backend)

    return struct.map(single_read, names)


def write_sim_frame(directory: str or tuple or list,
                    fields: field.Field or tuple or list or dict or struct.Struct,
                    frame: int,
                    names: str or tuple or list or struct.Struct or None = None,
                    batch_dim: str = 'batch'):
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
        batch_dim: (Optional) Only for batch write. Fields are unstacked along this dimension and matched with the list of directories.
    """
    assert isinstance(directory, (str, tuple, list))
    batch_mode = isinstance(directory, (tuple, list))
    if names is None:
        names = struct.names(fields)
    if frame > 1000000:
        warnings.warn(f"frame too large: {frame}. Data will be saved but filename might cause trouble in the future.")

    def single_write(f, name):
        name = _slugify_filename(name)
        if isinstance(f, field.SampledField):
            if batch_mode:
                f = f.unstack(batch_dim)
                assert len(f) == len(directory), f"The batch size '{batch_dim}' of '%{f}' is {len(f)} and does not match the number of directories, {len(directory)}"
                for f_, dir_ in zip(f, directory):
                    field.write(f_, _filename(dir_, name, frame))
            else:
                field.write(f, _filename(directory, name, frame))
        elif isinstance(f, math.Tensor):
            raise NotImplementedError()
        elif isinstance(f, field.Field):
            raise ValueError("write_sim_frame: only SampledField instances are saved. Resample other Fields before saving them.")
        else:
            raise ValueError(f"write_sim_frame: only SampledField instances can be saved but got {f}")

    struct.foreach(single_write, fields, names)


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
    single_fieldname = isinstance(fieldnames, str)
    if single_fieldname:
        fieldnames = [fieldnames]

    field_lists = [[] for f in fieldnames]
    for i in frames:
        fields = list(read_sim_frame(simpath, fieldnames, i, set_missing_to_none=False))
        for j in range(len(fieldnames)):
            field_lists[j].append(fields[j])
    result = [np.concatenate(list, 0) for list in field_lists]
    return result if not single_fieldname else result[0]


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

    def __init__(self,
                 parent_directory: str,
                 scene_id: int or list,
                 path: str or list,
                 batch_dim: str or None = None):
        self.parent_directory: str = parent_directory
        """ Directory containing the scene directory. """
        self.id = scene_id
        """
        All scenes inside one directory have a unique id.
        The id of a new Scene is always 1 + previously largest id.
        
        For a single scene, `type(id) = int`, for a batch, `type(id) = tuple`.
        """
        self._properties = None
        self.path = path
        """
        Relative path of the scene directory.
        
        For a single scene, `type(path) = str`, for a batch, `type(path) = tuple`.
        """
        if isinstance(path, str):
            self.abs_path = abspath(path)
        else:
            self.abs_path = tuple(abspath(p) for p in path)
        """
        Absolute path of the scene directory.
        
        For a single scene, `type(abs_path) = str`, for a batch, `type(abs_path) = tuple`.
        """
        self.batch_dim = batch_dim

    @staticmethod
    def create(parent_directory, count=1, copy_calling_script=True, batch_dim: str or None = None) -> 'Scene':
        """
        Creates a new `Scene` or a batch of new scenes inside `parent_directory`.

        See Also:
            `Scene.at()`, `Scene.list()`.

        Args:
            parent_directory: Directory to hold the new `Scene`. If it doesn't exist, it will be created.
            count: Number of scenes to create. Multiple scenes will also also represented by a single `Scene` object.
            copy_calling_script: Whether to copy the Python file that invoked this method into the `src` folder of all created scenes.
                See `Scene.copy_calling_script()`.
            batch_dim: Dimension corresponding to batch of scenes if `count > 1`.

        Returns:
            Single `Scene` object representing the new scene(s).
        """
        if count > 1:
            assert isinstance(batch_dim, str), "batch_dim must be specified if count > 1."
            scenes = [Scene.create(parent_directory, 1, copy_calling_script) for _ in range(count)]
            return Scene(parent_directory, tuple(s.id for s in scenes), tuple(s.path for s in scenes), batch_dim=batch_dim)
        else:
            parent_directory = expanduser(parent_directory)
            abs_dir = abspath(parent_directory)
            if not isdir(abs_dir):
                os.makedirs(abs_dir)
                next_id = 0
            else:
                indices = [int(name[4:]) for name in os.listdir(abs_dir) if name.startswith("sim_")]
                next_id = max([-1] + indices) + 1
            path = join(parent_directory, f"sim_{next_id:06d}")
            scene = Scene(parent_directory, next_id, path)
            scene.mkdir()
            if copy_calling_script:
                try:
                    scene.copy_calling_script()
                except IOError as err:
                    warnings.warn(f"Failed to copy calling script to scene during Scene.create(): {err}")
            return scene

    @staticmethod
    def list(parent_directory: str, include_other: bool = False) -> tuple:
        """
        Lists all scenes inside the given directory.

        See Also:
            `Scene.at()`, `Scene.create()`.

        Args:
            parent_directory: Directory that contains scene folders.
            include_other: Whether folders that do not match the scene format should also be treated as scenes.

        Returns:
            `tuple` of scenes.
        """
        parent_directory = expanduser(parent_directory)
        abs_dir = abspath(parent_directory)
        if not isdir(abs_dir):
            return ()
        names = [sim for sim in os.listdir(abs_dir) if sim.startswith("sim_") or (include_other and isdir(join(abs_dir, sim)))]
        return tuple(Scene(parent_directory, scene_id=int(name[4:]) if name.startswith("sim_") else None, path=join(parent_directory, name))
                     for name in names)

    @staticmethod
    def at(directory, id: int or None = None) -> 'Scene':
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
        directory = expanduser(directory)
        if id is not None:
            assert not basename(directory).startswith('sim_'), "When passing a scene directory, set id=None."
            path = join(directory, f"sim_{id:06d}")
            if not isdir(path):
                raise IOError(f"There is no scene at '{path}'")
            return Scene(directory, id, path)
        else:
            if directory[-1] == '/':  # remove trailing backslash
                directory = directory[0:-1]
            if not isdir(directory):
                raise IOError(f"There is no scene at '{abspath(directory)}'")
            scene_name = basename(directory)
            scene_id = int(scene_name[4:]) if scene_name.startswith("sim_") else None
            parent_directory = dirname(directory)
            return Scene(parent_directory, scene_id, directory)

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
        if isinstance(self.path, str):
            path = join(self.path, name)
            if create and not isdir(path):
                os.mkdir(path)
            return path
        else:
            result = []
            for p in self.path:
                path = join(p, name)
                if create and not isdir(path):
                    os.mkdir(path)
                result.append(path)
            return tuple(result)

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

    # def read_sim_frames(self, fieldnames=None, frames=None):
    #     return read_sim_frames(self.path, fieldnames=fieldnames, frames=frames, batch_dim=self.batch_dim)

    def read_array(self, field_name, frame):
        return read_sim_frame(self.path, field_name, frame=frame, batch_dim=self.batch_dim)

    def write_sim_frame(self, arrays, fieldnames, frame):
        write_sim_frame(self.path, arrays, names=fieldnames, frame=frame, batch_dim=self.batch_dim)

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
        write_sim_frame(self.path, data, names=None, frame=frame, batch_dim=self.batch_dim)

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
        return read_sim_frame(self.path, names, frame=frame, convert_to_backend=convert_to_backend, batch_dim=self.batch_dim)

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

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path

    def __eq__(self, other):
        return isinstance(other, Scene) and other.path == self.path

    def copy_calling_script(self, full_trace=False, include_context_information=True):
        """
        Copies the Python file that called this method into the `src` folder of this `Scene`.

        Args:
            full_trace: Whether to include scripts that indirectly called this method.
            include_context_information: If True, writes the phiflow version and `sys.argv` into `context.json`.
        """
        script_paths = [frame[1] for frame in inspect.stack()]
        script_paths = list(filter(lambda path: not _is_phi_file(path), script_paths))
        script_paths = set(script_paths) if full_trace else [script_paths[0]]
        for script_path in script_paths:
            shutil.copy(script_path, join(self.subpath('src', create=True), basename(script_path)))
        if include_context_information:
            with open(join(self.subpath('src', create=True), 'context.json'), 'w') as context_file:
                json.dump({
                    'phi_version': phi_version,
                    'argv': sys.argv
                }, context_file)

    def copy_src(self, path, only_external=True):
        if not only_external or not _is_phi_file(path):
            shutil.copy(path, join(self.subpath('src', create=True), basename(path)))

    def mkdir(self, subdir=None):
        assert isinstance(self.path, str)
        isdir(self.path) or os.mkdir(self.path)
        if subdir is not None:
            subpath = join(self.path, subdir)
            isdir(subpath) or os.mkdir(subpath)

    def remove(self):
        """ Deletes the scene directory and all contained files. """
        if isinstance(self.abs_path, str):
            if isdir(self.abs_path):
                shutil.rmtree(self.abs_path)
        else:
            for p in self.abs_path:
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
