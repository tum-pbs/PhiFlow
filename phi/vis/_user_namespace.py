import inspect
import os
import sys


class UserNamespace:

    def list_variables(self, only_public=False, only_current_scope=False) -> dict:
        raise NotImplementedError(self)

    def get_variable(self, name: str, default=None):
        raise NotImplementedError(self)

    def set_variable(self, name: str, value):
        raise NotImplementedError(self)

    def get_title(self):
        raise NotImplementedError(self)

    def get_description(self):
        raise NotImplementedError(self)

    def get_reference(self):
        raise NotImplementedError(self)


def default_user_namespace(call_stack_depth=1) -> UserNamespace:
    jupyter = 'ipykernel' in sys.modules
    return JupyterNamespace() if jupyter else ModuleNamespace(call_stack_depth + 1)


class ModuleNamespace(UserNamespace):

    def __init__(self, call_stack_depth: int):
        self.module = inspect.getmodule(inspect.stack()[call_stack_depth + 1].frame)

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
        if only_public:
            return {name: getattr(self.module, name) for name in dir(self.module) if not name.startswith('_')}
        else:
            return {name: getattr(self.module, name) for name in dir(self.module)}

    def get_variable(self, name: str, default=None):
        return getattr(self.module, name, default)

    def set_variable(self, name: str, value):
        setattr(self.module, name, value)


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

