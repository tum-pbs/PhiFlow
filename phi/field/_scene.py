import inspect
import json
import os
import re
import shutil
import sys
import warnings
from os.path import join, isfile, isdir, abspath, expanduser, basename, split

from phi import math, __version__ as phi_version
from ._field import SampledField
from ._field_io import read, write
from ..math import Shape, batch, stack, unpack_dim, wrap
from ..math.magic import BoundDim


def _filename(simpath, name, frame):
    return join(simpath, f"{slugify(name)}_{frame:06d}.npz")


def _str(bytes_or_str):  # on Linux, os.listdir returns bytes instead of strings
    if isinstance(bytes_or_str, str):
        return bytes_or_str
    else:
        return str(bytes_or_str, 'utf-8')


def get_fieldnames(simpath) -> tuple:
    fieldnames_set = {_str(f)[:-11] for f in os.listdir(simpath) if _str(f).endswith(".npz")}
    return tuple(sorted(fieldnames_set))


def get_frames(path: str, field_name: str = None, mode=set.intersection) -> tuple:
    if field_name is not None:
        all_frames = {int(f[-10:-4]) for f in os.listdir(path) if _str(f).startswith(field_name) and _str(f).endswith(".npz")}
        return tuple(sorted(all_frames))
    else:
        fields = get_fieldnames(path)
        if not fields:
            return ()
        frames_sets = [set(get_frames(path, field)) for field in fields]
        frames = mode(*frames_sets)
        return tuple(sorted(frames))


class Scene:
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
        self._paths = math.wrap(paths)
        self._properties: dict or None = None

    def __getitem__(self, item):
        return Scene(self._paths[item])

    def __getattr__(self, name: str) -> BoundDim:
        return BoundDim(self, name)

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Scene':
        if all(isinstance(v, Scene) for v in values):
            return Scene(stack([v.paths for v in values], dim, **kwargs))
        else:
            return NotImplemented

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
    def stack(*scenes: 'Scene', dim: Shape = batch('batch')) -> 'Scene':
        return Scene(math.stack([s._paths for s in scenes], dim))

    @staticmethod
    def create(parent_directory: str,
               shape: math.Shape = math.EMPTY_SHAPE,
               name='sim',
               copy_calling_script=True,
               **dimensions) -> 'Scene':
        """
        Creates a new `Scene` or a batch of new scenes inside `parent_directory`.

        See Also:
            `Scene.at()`, `Scene.list()`.

        Args:
            parent_directory: Directory to hold the new `Scene`. If it doesn't exist, it will be created.
            shape: Determines number of scenes to create. Multiple scenes will be represented by a `Scene` with `is_batch=True`.
            name: Name of the directory (excluding index). Default is `'sim'`.
            copy_calling_script: Whether to copy the Python file that invoked this method into the `src` folder of all created scenes.
                See `Scene.copy_calling_script()`.
            dimensions: Additional batch dimensions

        Returns:
            Single `Scene` object representing the new scene(s).
        """
        shape = shape & math.batch(**dimensions)
        parent_directory = expanduser(parent_directory)
        abs_dir = abspath(parent_directory)
        if not isdir(abs_dir):
            os.makedirs(abs_dir)
            next_id = 0
        else:
            indices = [int(f[len(name)+1:]) for f in os.listdir(abs_dir) if f.startswith(f"{name}_")]
            next_id = max([-1] + indices) + 1
        ids = unpack_dim(wrap(tuple(range(next_id, next_id + shape.volume))), 'vector', shape)
        paths = math.map(lambda id_: join(parent_directory, f"{name}_{id_:06d}"), ids)
        scene = Scene(paths)
        scene.mkdir()
        if copy_calling_script:
            try:
                scene.copy_calling_script()
            except IOError as err:
                warnings.warn(f"Failed to copy calling script to scene during Scene.create(): {err}", RuntimeWarning)
        return scene

    @staticmethod
    def list(parent_directory: str,
             name='sim',
             include_other: bool = False,
             dim: Shape or None = None) -> 'Scene' or tuple:
        """
        Lists all scenes inside the given directory.

        See Also:
            `Scene.at()`, `Scene.create()`.

        Args:
            parent_directory: Directory that contains scene folders.
            name: Name of the directory (excluding index). Default is `'sim'`.
            include_other: Whether folders that do not match the scene format should also be treated as scenes.
            dim: Stack dimension. If None, returns tuple of `Scene` objects. Otherwise, returns a scene batch with this dimension.

        Returns:
            `tuple` of scenes.
        """
        parent_directory = expanduser(parent_directory)
        abs_dir = abspath(parent_directory)
        if not isdir(abs_dir):
            return ()
        names = [sim for sim in os.listdir(abs_dir) if sim.startswith(f"{name}_") or (include_other and isdir(join(abs_dir, sim)))]
        if dim is None:
            return tuple(Scene(join(parent_directory, n)) for n in names)
        else:
            paths = math.wrap([join(parent_directory, n) for n in names], dim)
            return Scene(paths)

    @staticmethod
    def at(directory: str or tuple or list or math.Tensor or 'Scene', id: int or math.Tensor or None = None) -> 'Scene':
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
        if isinstance(directory, Scene):
            assert id is None, f"Got id={id} but directory is already a Scene."
            return directory
        if isinstance(directory, (tuple, list)):
            directory = math.wrap(directory, batch('scenes'))
        directory = math.map(lambda d: expanduser(d), math.wrap(directory))
        if id is None:
            paths = directory
        else:
            id = math.wrap(id)
            paths = math.map(lambda d, i: join(d, f"sim_{i:06d}"), directory, id)
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

        def read_json(path: str) -> dict:
            json_file = join(path, "description.json")
            if isfile(json_file):
                with open(json_file) as stream:
                    props = json.load(stream)
                if '__tensors__' in props:
                    for key in props['__tensors__']:
                        props[key] = math.from_dict(props[key])
                return props
            else:
                return {}

        if self._paths.shape.volume == 1:
            self._properties = read_json(self._paths.native())
        else:
            self._properties = {}
            dicts = [read_json(p) for p in self._paths]
            keys = set(sum([tuple(d.keys()) for d in dicts], ()))
            for key in keys:
                assert all(key in d for d in dicts), f"Failed to create batched Scene because property '{key}' is present in some scenes but not all."
                if all([math.all(d[key] == dicts[0][key]) for d in dicts]):
                    self._properties[key] = dicts[0][key]
                else:
                    self._properties[key] = stack([d[key] for d in dicts], self._paths.shape)
        if '__tensors__' in self._properties:
            del self._properties['__tensors__']

    def exist_properties(self):
        """
        Checks whether the file `description.json` exists or has existed.
        """
        if self._properties is not None:
            return True  # must have been written or read
        else:
            json_file = join(next(iter(math.flatten(self._paths))), "description.json")
            return isfile(json_file)

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
        """ See `Scene.put_properties()`. """
        self._init_properties()
        self._properties[key] = value
        self._write_properties()

    def put_properties(self, update: dict = None, **kw_updates):
        """
        Updates the properties dictionary and stores it in `description.json` of all scene folders.

        Args:
            update: new values, must be JSON serializable.
            kw_updates: additional update as keyword arguments. This overrides `update`.
        """
        self._init_properties()
        if update:
            self._properties.update(update)
        self._properties.update(kw_updates)
        self._write_properties()

    def _get_properties(self, index: dict):
        result = dict(self._properties)
        tensor_names = []
        for key, value in self._properties.items():
            if isinstance(value, math.Tensor):
                value = value[index]
                if value.rank == 0:
                    value = value.dtype.kind(value)
                else:
                    value = math.to_dict(value)
                    tensor_names.append(key)
                result[key] = value
        if tensor_names:
            result['__tensors__'] = tuple(tensor_names)
        return result

    def _write_properties(self):
        for instance in self.paths.shape.meshgrid():
            path = self.paths[instance].native()
            instance_properties = self._get_properties(instance)
            with open(join(path, "description.json"), "w") as out:
                json.dump(instance_properties, out, indent=2)

    def write(self, data: dict = None, frame=0, **kw_data):
        """
        Writes fields to this scene.
        One NumPy file will be created for each `phi.field.Field`

        See Also:
            `Scene.read()`.

        Args:
            data: `dict` mapping field names to `Field` objects that can be written using `phi.field.write()`.
            kw_data: Additional data, overrides elements in `data`.
            frame: Frame number.
        """
        data = dict(data) if data else {}
        data.update(kw_data)
        for name, field in data.items():
            self.write_field(field, name, frame)

    def write_field(self, field: SampledField, name: str, frame: int):
        """
        Write a `SampledField` to a file.
        The filenames are created from the provided names and the frame index in accordance with the
        scene format specification at https://tum-pbs.github.io/PhiFlow/Scene_Format_Specification.html .

        Args:
            field: single field or structure of Fields to save.
            name: Base file name.
            frame: Frame number as `int`, typically time step index.
        """
        if not isinstance(field, SampledField):
            raise ValueError(f"Only SampledField instances can be saved but got {field}")
        name = _slugify_filename(name)
        files = math.map(lambda dir_: _filename(dir_, name, frame), self._paths)
        write(field, files)

    def read_field(self, name: str, frame: int, convert_to_backend=True) -> SampledField:
        """
        Reads a single `SampledField` from files contained in this `Scene` (batch).

        Args:
            name: Base file name.
            frame: Frame number as `int`, typically time step index.
            convert_to_backend: Whether to convert the read data to the data format of the default backend, e.g. TensorFlow tensors.

        Returns:
            `SampledField`
        """
        name = _slugify_filename(name)
        files = math.map(lambda dir_: _filename(dir_, name, frame), self._paths)
        return read(files, convert_to_backend=convert_to_backend)

    read_array = read_field

    def read(self, *names: str, frame=0, convert_to_backend=True):
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
        if len(names) == 1 and isinstance(names[0], (tuple, list)):
            names = names[0]
        result = [self.read_array(name, frame, convert_to_backend) for name in names]
        return result[0] if len(names) == 1 else result

    @property
    def fieldnames(self) -> tuple:
        """ Determines all field names present in this `Scene`, independent of frame. """
        return get_fieldnames(self.path)

    @property
    def frames(self):
        """ Determines all frame numbers present in this `Scene`, independent of field names. See `Scene.complete_frames`. """
        return get_frames(self.path, mode=set.union)

    @property
    def complete_frames(self):
        """
        Determines all frame number for which all existing fields are available.
        If there are multiple fields stored within this scene, a frame is considered complete only if an entry exists for all fields.

        See Also:
            `Scene.frames`
        """
        return get_frames(self.path, mode=set.intersection)

    def __repr__(self):
        return f"{self.paths:no-dtype}"

    def __eq__(self, other):
        return isinstance(other, Scene) and (other._paths == self._paths).all

    def copy_calling_script(self, full_trace=False, include_context_information=True):
        """
        Copies the Python file that called this method into the `src` folder of this `Scene`.

        In batch mode, the script is copied to all scenes.

        Args:
            full_trace: Whether to include scripts that indirectly called this method.
            include_context_information: If True, writes the phiflow version and `sys.argv` into `context.json`.
        """
        script_paths = [frame.filename for frame in inspect.stack()]
        script_paths = list(filter(lambda path: not _is_phi_file(path), script_paths))
        script_paths = set(script_paths) if full_trace else [script_paths[0]]
        self.subpath('src', create=True)
        for script_path in script_paths:
            if script_path.endswith('.py'):
                self.copy_src(script_path, only_external=False)
            elif 'ipython' in script_path:
                from IPython import get_ipython
                cells = get_ipython().user_ns['In']
                blocks = [f"#%% In[{i}]\n{cell}" for i, cell in enumerate(cells)]
                text = "\n\n".join(blocks)
                self.copy_src_text('ipython.py', text)
        if include_context_information:
            for path in math.flatten(self._paths):
                with open(join(path, 'src', 'context.json'), 'w') as context_file:
                    json.dump({
                        'phi_version': phi_version,
                        'argv': sys.argv
                    }, context_file)

    def copy_src(self, script_path, only_external=True):
        for path in math.flatten(self._paths):
            if not only_external or not _is_phi_file(script_path):
                shutil.copy(script_path, join(path, 'src', basename(script_path)))

    def copy_src_text(self, filename, text):
        for path in math.flatten(self._paths):
            target = join(path, 'src', filename)
            with open(target, "w") as file:
                file.writelines(text)

    def mkdir(self):
        for path in math.flatten(self._paths):
            isdir(path) or os.mkdir(path)

    def remove(self):
        """ Deletes the scene directory and all contained files. """
        for p in math.flatten(self._paths):
            p = abspath(p)
            if isdir(p):
                shutil.rmtree(p)


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
    value = re.sub('[^\\w\\s-]', '', value).strip().lower()
    value = re.sub('[-\\s]+', '-', value)
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
