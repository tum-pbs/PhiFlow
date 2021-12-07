import inspect
import os
import sys
from typing import List

from phi.vis._vis_base import display_name


class UserNamespace:

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        raise NotImplementedError(self)

    def get_variable(self, name: str, default=None):  # __getitem__
        raise NotImplementedError(self)

    def set_variable(self, name: str, value):  # __setitem__
        raise NotImplementedError(self)

    def get_title(self):  # __repr__()
        raise NotImplementedError(self)

    def get_description(self):  # __doc__?
        raise NotImplementedError(self)

    def get_reference(self):  # __str__() / __repr__()
        """ Used to determine the default directory name to which data is written. """
        raise NotImplementedError(self)


def global_user_namespace(frames: List[inspect.FrameInfo]) -> UserNamespace:
    if 'ipykernel' in sys.modules:
        return JupyterNamespace()
    else:
        for frame in frames:
            if frame.function == '<module>':
                module = inspect.getmodule(frame.frame)
                return ModuleNamespace(module)
        raise AssertionError('No module found in call stack.')


def get_user_namespace(ignore_stack_frames=0, frames: List[inspect.FrameInfo] = None) -> UserNamespace:
    if not frames:
        frames = inspect.stack()[ignore_stack_frames + 1:]  # 1 for this function
    if frames[0].function == '<module>':
        return global_user_namespace(frames)
    else:
        return LocalNamespace(frames)


class ModuleNamespace(UserNamespace):

    def __init__(self, module):
        self.module = module

    def get_title(self):
        doc = self.module.__doc__
        if doc is None:
            return self.get_reference()
        else:
            end_of_line = doc.index('\n')
            name = doc[:end_of_line].strip()
            return name if name else self.get_reference()

    def get_reference(self):
        return os.path.basename(self.module.__file__)[:-3]

    def get_description(self):
        doc = self.module.__doc__
        if doc is None:
            return doc or self.module.__file__
        else:
            end_of_line = doc.index('\n')
            return doc[end_of_line:].strip() or ""

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        # all variables are in the current scope
        variables = {name: getattr(self.module, name) for name in dir(self.module)}
        if only_public:
            variables = {n: v for n, v in variables.items() if not n.startswith('_')}
        if only_current_scope:
            variables = {n: v for n, v in variables.items() if is_value(v) or inspect.getmodule(v) == self.module}
        return variables

    def get_variable(self, name: str, default=None):
        return getattr(self.module, name, default)

    def set_variable(self, name: str, value):
        setattr(self.module, name, value)


def is_value(obj):
    if isinstance(obj, type):
        return False
    # return type(open).__name__ not in ('function', 'builtin_function_or_method', 'module')
    # if isinstance(obj, (type, function, builtin_function_or_method))
    if inspect.isfunction(obj):
        return False
    return True


class JupyterNamespace(UserNamespace):

    def get_title(self):
        return "Notebook"

    def get_reference(self):
        return "notebooks"

    def get_description(self):
        if 'google.colab' not in sys.modules:
            return "Google Colab Notebook"
        else:
            import ipykernel
            version = ipykernel.version_info
            client_name = ipykernel.write_connection_file.__module__.split('.')[0]
            return f"{client_name} ({version})"

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        from IPython import get_ipython
        shell = get_ipython()
        variables = shell.user_ns
        if only_public:
            hidden = shell.user_ns_hidden
            variables = {n: v for n, v in variables.items() if not n.startswith('_') and n not in hidden}
        if only_current_scope:
            text = shell.user_ns['In'][-1]
            variables = {n: v for n, v in variables.items() if n in text}  # TODO parse text, only show names with assignment
        return variables

    def get_variable(self, name: str, default=None):
        from IPython import get_ipython
        shell = get_ipython()
        return shell.user_ns.get(name, default)

    def set_variable(self, name: str, value):
        from IPython import get_ipython
        shell = get_ipython()
        shell.user_ns[name] = value


class LocalNamespace(UserNamespace):

    def __init__(self, frames: List[inspect.FrameInfo]):
        self.frame = frames[0].frame
        self.function_name: str = frames[0].function
        self.module = inspect.getmodule(self.frame)
        if self.module:
            if hasattr(self.module, self.function_name):
                self.function = getattr(self.module, self.function_name)
            else:
                self.function = None
        else:
            assert 'ipykernel' in sys.modules, f"Unable to locate file in which {self.function_name} is declared."
            self.function = JupyterNamespace().get_variable(self.function_name)

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        return self.frame.f_locals

    def get_variable(self, name: str, default=None):
        return self.frame.f_locals.get(name, default)

    def set_variable(self, name: str, value):
        import ctypes
        self.frame.f_locals[name] = value
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(self.frame), ctypes.c_int(0))

    def get_title(self):
        return display_name(self.function_name)

    def get_description(self):
        if self.function is not None and self.function.__doc__:
            return self.function.__doc__
        else:
            return f"Function `{self.function_name}()` in '{self.module.__file__}'"

    def get_reference(self):
        return f"{os.path.basename(self.module.__file__)[:-3]}_{self.function_name}"


class DictNamespace(UserNamespace):

    def __init__(self,
                 variables: dict,
                 title: str = "Unknown",
                 description: str = "Custom namespace, unknown source.",
                 reference: str = 'unknown'):
        self.variables = variables
        self.title = title
        self.description = description
        self.reference = reference

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        variables = self.variables
        if only_public:
            variables = {n: v for n, v in variables.items() if not n.startswith('_')}
        return variables

    def get_variable(self, name: str, default=None):
        return self.variables.get(name, default)

    def set_variable(self, name: str, value):
        self.variables[name] = value

    def get_title(self):
        return self.title

    def get_description(self):
        return self.description

    def get_reference(self):
        return self.reference
