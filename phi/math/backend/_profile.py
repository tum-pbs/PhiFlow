import inspect, traceback
import json
import sys
from contextlib import contextmanager
from time import perf_counter

from ._backend import Backend


class BackendCall:

    def __init__(self, start: float, stop: float, backend: 'ProfilingBackend', function_name):
        self._start = start
        self._stop = stop
        self._backend = backend
        self._function_name = function_name

    def __repr__(self):
        return f"{1000 * self._duration:.2f} ms  {self._function_name}"

    def print(self, include_parents, depth, min_duration, code_col, code_len):
        if self._duration >= min_duration:
            print(f"{'  ' * depth}{1000 * self._duration:.2f} ms  {self._backend}.{self._function_name}")

    @property
    def _name(self):
        return repr(self)

    @property
    def _duration(self):
        return self._stop - self._start

    def trace_json_events(self, include_parents) -> list:
        backend_index = self._backend._index
        name = f"{self._backend.name}.{self._function_name}"
        return [
            {"name": name, "cat": "CALL", "ph": "B", "pid": 1, "tid": backend_index, "ts": int(round(self._start * 1000000))},  # Begin
            {"name": name, "cat": "CALL", "ph": "E", "pid": 1, "tid": backend_index, "ts": int(round(self._stop * 1000000))}  # End
        ]

    def call_count(self) -> int:
        return 1


class ExtCall:

    def __init__(self, parent: 'ExtCall' or None, stack: list):
        self._parent = parent
        if parent is None:
            self._parents = ()
        else:
            self._parents = parent._parents + (parent,)
        self._stack = stack  # stack trace from inspect.stack() including parent calls
        self._children = []  # BackendCalls and ExtCalls
        self._converted = False

    def common_call(self, stack: list):
        """ Returns the deepest ExtCall in the hierarchy of this call that contains `stack`. """
        if self._parent is None:
            return self
        if len(stack) < len(self._stack):
            return self._parent.common_call(stack)
        for i in range(len(self._stack)):
            if self._stack[-1-i].function != stack[-1-i].function:
                return self._parents[i]
        return self

    def add(self, child):
        self._children.append(child)

    @property
    def _name(self):
        if not self._stack:
            return ""
        info = self._stack[0]
        fun = info.function
        if 'self' in info.frame.f_locals:
            if fun == '__init__':
                return f"{type(info.frame.f_locals['self']).__name__}()"
            return f"{type(info.frame.f_locals['self']).__name__}.{fun}"
        if 'phi/math' in info.filename or 'phi\\math' in info.filename:
            return f"math.{fun}"
        else:
            return fun

    @property
    def _start(self):
        return self._children[0]._start

    @property
    def _stop(self):
        return self._children[-1]._stop

    @property
    def _duration(self):
        return sum(c._duration for c in self._children)

    def call_count(self) -> int:
        return sum(child.call_count() for child in self._children)

    def __repr__(self):
        if not self._converted:
            if self._parent is None:
                return "/"
            return f"{self._name} ({len(self._stack)})"
        else:
            context = self._stack[0].code_context
            return f"sum {1000 * self._duration:.2f} ms  {context}"

    def __len__(self):
        return len(self._children)

    def print(self, include_parents=(), depth=0, min_duration=0., code_col=80, code_len=50):
        if self._duration < min_duration:
            return
        if len(self._children) == 1 and isinstance(self._children[0], ExtCall):
            self._children[0].print(include_parents + ((self,) if self._parent is not None else ()), depth, min_duration, code_col, code_len)
        else:
            funcs = [par._name for par in include_parents] + [self._name]
            text = f"{'. ' * depth}-> {' -> '.join(funcs)} ({1000 * self._duration:.2f} ms)"
            if len(self._stack) > len(include_parents)+1:
                code = self._stack[len(include_parents)+1].code_context[0].strip()
                if len(code) > code_len:
                    code = code[:code_len-3] + "..."
                text += " " + "." * max(0, (code_col - len(text))) + " > " + code
            print(text)
            for child in self._children:
                child.print((), depth + 1, min_duration, code_col, code_len)

    def children_to_properties(self) -> dict:
        result = {}
        for child in self._children:
            name = f"{len(result)} {child._name}" if len(self._children) <= 10 else f"{len(result):02d} {child._name}"
            while isinstance(child, ExtCall) and len(child) == 1:
                child = child._children[0]
                name += " -> " + child._name
            result[name] = child
            if isinstance(child, ExtCall):
                child.children_to_properties()
        # finalize
        for name, child in result.items():
            setattr(self, name, child)
        self._converted = True
        return result

    def trace_json_events(self, include_parents=()) -> list:
        if len(self._children) == 1:
            return self._children[0].trace_json_events(include_parents + (self,))
        else:
            name = ' -> '.join([par._name for par in include_parents] + [self._name]) + f" ({self.call_count()} calls)"
            result = [
                {"name": name, "cat": "METHOD", "ph": "B", "pid": 0, "tid": 0, "ts": int(round(self._start * 1000000))},  # Begin
                {"name": name, "cat": "METHOD", "ph": "E", "pid": 0, "tid": 0, "ts": int(round(self._stop * 1000000))}  # End
            ]
            for child in self._children:
                result.extend(child.trace_json_events(()))
            return result


class Profile:

    def __init__(self):
        self._start = perf_counter()
        self._stop = None
        self._root = ExtCall(None, [])
        self._last_ext_call = self._root
        self._messages = []

    def add_call(self, backend_call: BackendCall):
        stack = inspect.stack()[2:]
        call = self._last_ext_call.common_call(stack)
        for i in range(len(call._stack), len(stack)):
            sub_call = ExtCall(call, stack[len(stack) - i - 1:])
            call.add(sub_call)
            call = sub_call
        call.add(backend_call)
        self._last_ext_call = call

    def stop(self):
        self._stop = perf_counter()
        self._children_to_properties()

    @property
    def duration(self):
        return self._stop - self._start if self._stop is not None else None

    def print(self, min_duration=1e-3, code_col=80, code_len=50):
        print(f"Profile: {self.duration:.4f} seconds total. Skipping elements shorter than {1000 * min_duration:.2f} ms")
        if self._messages:
            print("External profiling:")
            for message in self._messages:
                print(f"  {message}")
            print()
        self._root.print(min_duration=min_duration, code_col=code_col, code_len=code_len)

    def save_trace(self, json_file):
        if len(self._root._children) == 0:
            return
        data = self._root.trace_json_events()
        with open(json_file, 'w') as file:
            json.dump(data, file)

    def _children_to_properties(self):
        children = self._root.children_to_properties()
        for name, child in children.items():
            setattr(self, name, child)

    def add_external_message(self, message):
        self._messages.append(message)


class ProfilingBackend:

    def __init__(self, prof: Profile, backend: Backend, index: int):
        self._backend = backend
        self._profile = prof
        self._index = index
        # non-profiling methods
        self.name = backend.name
        self.combine_types = backend.combine_types
        self.auto_cast = backend.auto_cast
        self.matches_name = backend.matches_name
        self.is_tensor = backend.is_tensor
        self.is_available = backend.is_available
        self.shape = backend.shape
        self.staticshape = backend.staticshape
        self.ndims = backend.ndims
        # profiling methods
        for item_name in dir(backend):
            item = getattr(backend, item_name)
            if callable(item) and not hasattr(self, item_name):
                def context(item=item, item_name=item_name, profiling_backend=self):
                    def call_fun(*args, **kwargs):
                        start = perf_counter()
                        result = item(*args, **kwargs)
                        stop = perf_counter()
                        prof.add_call(BackendCall(start, stop, profiling_backend, item_name))
                        return result
                    return call_fun
                setattr(self, item_name, context())

    def __repr__(self):
        return f"profile[{self._backend}]"


_PROFILE = []


@contextmanager
def profile(backends=None):
    prof = Profile()
    _PROFILE.append(prof)

    # Replace backends
    from . import BACKENDS, _DEFAULT
    original_backends = tuple(BACKENDS)
    backends = original_backends if backends is None else backends
    for i, backend in enumerate(backends):
        prof_backend = ProfilingBackend(prof, backend, i)
        BACKENDS[BACKENDS.index(backend)] = prof_backend
        if _DEFAULT[-1] == backend:
            _DEFAULT[-1] = prof_backend

    try:
        yield prof
    finally:
        prof.stop()
        BACKENDS.clear()
        BACKENDS.extend(original_backends)
        _PROFILE.pop(-1)
        # print(f'Profiling session lasted for {1000 * (perf_counter() - start)} ms')


def get_current_profile():
    return _PROFILE[-1] if _PROFILE else None
