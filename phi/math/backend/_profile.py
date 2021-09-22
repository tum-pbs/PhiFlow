import inspect
import json
from contextlib import contextmanager
from time import perf_counter
from typing import Optional, Callable

from ._backend import Backend, BACKENDS, _DEFAULT


class BackendCall:

    def __init__(self, start: float, stop: float, backend: 'ProfilingBackend', function_name):
        self._start = start
        self._stop = stop
        self._backend = backend
        self._function_name = function_name
        self._args = {"Backend": backend.name}

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
        name = self._function_name
        return [
            {
                'name': name,
                'ph': 'X',
                'pid': 1,
                'tid': backend_index+1,
                'ts': int(round(self._start * 1000000)),
                'dur': int(round((self._stop - self._start) * 1000000)),
                'args': self._args
            }
        ]

    def call_count(self) -> int:
        return 1

    def add_arg(self, key, value):
        assert key not in self._args
        self._args[key] = value


class ExtCall:
    """ Function invocation that is not a Backend method but internally calls Backend methods. """

    def __init__(self,
                 parent: 'ExtCall' or None,
                 name: str,
                 level: int,
                 function: str,
                 code_context: list or None,
                 file_name: str,
                 line_number: int):
        """
        Args:
            parent: Parent call.
            name: Name of this call, see `ExtCall.determine_name()`.
            level: Number of parent stack items including this one.
        """
        self._parent = parent
        if parent is None:
            self._parents = ()
        else:
            self._parents = parent._parents + (parent,)
        self._children = []  # BackendCalls and ExtCalls
        self._converted = False
        self._name = name
        self._level = level
        self._function = function
        self._code_context = code_context
        self._file_name = file_name
        self._line_number = line_number

    def common_call(self, stack: list):
        """ Returns the deepest ExtCall in the hierarchy of this call that contains `stack`. """
        if self._parent is None:
            return self
        if len(stack) < self._level:
            return self._parent.common_call(stack)
        for i in range(self._level - 1):
            if self._parents[i+1]._function != stack[-1-i].function:
                return self._parents[i]
        return self

    def add(self, child):
        self._children.append(child)

    @staticmethod
    def determine_name(info):
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
            return f"{self._name} ({self._level})"
        else:
            context = self._code_context
            return f"sum {1000 * self._duration:.2f} ms  {context}"

    def __len__(self):
        return len(self._children)

    def _empty_parent_count(self):
        for i, parent in enumerate(reversed(self._parents)):
            if len(parent._children) > 1:
                return i
        return len(self._parents)

    def _eff_parent_count(self):
        return len([p for p in self._parents if len(p._children) > 1])

    def _closest_non_trivial_parent(self):
        parent = self._parent
        while parent._parent is not None:
            if len(parent._children) > 1:
                return parent
            parent = parent._parent
        return parent

    def _calling_code(self, backtrack=0):
        if self._level > backtrack + 1:
            call: ExtCall = self._parents[-backtrack-1]
            return call._code_context[0].strip(), call._file_name, call._function, call._line_number
        else:
            return "", "", "", -1

    def print(self, include_parents=(), depth=0, min_duration=0., code_col=80, code_len=50):
        if self._duration < min_duration:
            return
        if len(self._children) == 1 and isinstance(self._children[0], ExtCall):
            self._children[0].print(include_parents + ((self,) if self._parent is not None else ()), depth, min_duration, code_col, code_len)
        else:
            funcs = [par._name for par in include_parents] + [self._name]
            text = f"{'. ' * depth}-> {' -> '.join(funcs)} ({1000 * self._duration:.2f} ms)"
            if self._level > len(include_parents)+1:
                code = self._calling_code(backtrack=len(include_parents))[0]
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
            name = ' -> '.join([par._name for par in include_parents] + [self._name])
            eff_parent_count = self._eff_parent_count()
            calling_code, calling_filename, calling_function, lineno = self._calling_code(backtrack=self._empty_parent_count())
            result = [
                {
                    'name': name,
                    'ph': "X",  # complete event
                    'pid': 0,
                    'tid': eff_parent_count,
                    'ts': int(self._start * 1000000),
                    'dur': int((self._stop - self._start) * 1000000),
                    'args': {
                        "Calling code snippet": calling_code,
                        "Called by": f"{calling_function}() in {calling_filename}, line {lineno}",
                        "Active time (backend calls)": f"{self._duration * 1000:.2f} ms ({round(100 * self._duration / self._closest_non_trivial_parent()._duration):.0f}% of parent, {100 * self._duration / (self._stop - self._start):.1f}% efficiency)",
                        "Backend calls": f"{self.call_count()} ({round(100 * self.call_count() / self._closest_non_trivial_parent().call_count()):.0f}% of parent)"
                    }
                }
            ]
            for child in self._children:
                result.extend(child.trace_json_events(()))
            return result


class Profile:
    """
    Stores information about calls to backends and their timing.

    Profile may be created through `profile()` or `profile_function()`.

    Profiles can be printed or saved to disc.
    """

    def __init__(self, trace: bool, backends: tuple or list, subtract_trace_time: bool):
        self._start = perf_counter()
        self._stop = None
        self._root = ExtCall(None, "", 0, "", "", "", -1)
        self._last_ext_call = self._root
        self._messages = []
        self._trace = trace
        self._backend_calls = []
        self._retime_index = -1
        self._accumulating = False
        self._backends = backends
        self._subtract_trace_time = subtract_trace_time
        self._total_trace_time = 0

    def _add_call(self, backend_call: BackendCall, args: tuple, kwargs: dict, result):
        if self._retime_index >= 0:
            prev_call = self._backend_calls[self._retime_index]
            assert prev_call._function_name == backend_call._function_name
            if self._accumulating:
                prev_call._start += backend_call._start
                prev_call._stop += backend_call._stop
            else:
                prev_call._start = backend_call._start
                prev_call._stop = backend_call._stop
            self._retime_index = (self._retime_index + 1) % len(self._backend_calls)
        else:
            self._backend_calls.append(backend_call)
            args = {i: arg for i, arg in enumerate(args)}
            args.update(kwargs)
            backend_call.add_arg("Inputs", _format_values(args, backend_call._backend))
            if isinstance(result, (tuple, list)):
                backend_call.add_arg("Outputs", _format_values({i: res for i, res in enumerate(result)}, backend_call._backend))
            else:
                backend_call.add_arg("Outputs", _format_values({0: result}, backend_call._backend))
            if self._trace:
                stack = inspect.stack()[2:]
                call = self._last_ext_call.common_call(stack)
                for i in range(call._level, len(stack)):
                    stack_frame = stack[len(stack) - i - 1]
                    name = ExtCall.determine_name(stack_frame)  # if len(stack) - i > 1 else ""
                    sub_call = ExtCall(call, name, i + 1, stack_frame.function, stack_frame.code_context, stack_frame.filename, stack_frame.lineno)
                    call.add(sub_call)
                    call = sub_call
                call.add(backend_call)
                self._last_ext_call = call
            if self._subtract_trace_time:
                delta_trace_time = perf_counter() - backend_call._stop
                backend_call._start -= self._total_trace_time
                backend_call._stop -= self._total_trace_time
                self._total_trace_time += delta_trace_time

    def _finish(self):
        self._stop = perf_counter()
        self._children_to_properties()

    @property
    def duration(self) -> float:
        """ Total time passed from creation of the profile to the end of the last operation. """
        return self._stop - self._start if self._stop is not None else None

    def print(self, min_duration=1e-3, code_col=80, code_len=50):
        """
        Prints this profile to the console.

        Args:
            min_duration: Hides elements with less time spent on backend calls than `min_duration` (seconds)
            code_col: Formatting option for where the context code is printed.
            code_len: Formatting option for cropping the context code
        """
        print(f"Profile: {self.duration:.4f} seconds total. Skipping elements shorter than {1000 * min_duration:.2f} ms")
        if self._messages:
            print("External profiling:")
            for message in self._messages:
                print(f"  {message}")
            print()
        self._root.print(min_duration=min_duration, code_col=code_col, code_len=code_len)

    def save(self, json_file: str):
        """
        Saves this profile to disc using the *trace event format* described at
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit

        This file can be viewed with external applications such as Google chrome.

        Args:
            json_file: filename
        """
        data = [
            {'name': "process_name", 'ph': 'M', 'pid': 0, 'tid': 0, "args": {"name": "0 Python calls"}},
            {'name': "process_name", 'ph': 'M', 'pid': 1, 'tid': 1, "args": {"name": "1 Operations"}},
        ] + [
            {'name': "thread_name", 'ph': 'M', 'pid': 1, 'tid': i + 1, "args": {"name": backend.name}}
            for i, backend in enumerate(self._backends)
        ]
        if self._trace:
            if len(self._root._children) > 0:
                data.extend(self._root.trace_json_events())
        else:
            data.extend(sum([call.trace_json_events(()) for call in self._backend_calls], []))
        with open(json_file, 'w') as file:
            json.dump(data, file)

    save_trace = save

    def _children_to_properties(self):
        children = self._root.children_to_properties()
        for name, child in children.items():
            setattr(self, name, child)

    def add_external_message(self, message: str):
        """ Stores an external message in this profile. External messages are printed in `Profile.print()`. """
        self._messages.append(message)

    @contextmanager
    def retime(self):
        """
        To be used in `with` statements, `with prof.retime(): ...`.

        Updates this profile by running the same operations again but without tracing.
        This gives a much better indication of the true timing.
        The code within the `with` block must perform the same operations as the code that created this profile.

        *Warning:* Internal caching may reduce the number of operations after the first time a function is called.
        To prevent this, run the function before profiling it, see `warmup` in `profile_function()`.
        """
        self._retime_index = 0
        restore_data = _start_profiling(self, self._backends)
        try:
            yield None
        finally:
            _stop_profiling(self, *restore_data)
            assert self._retime_index == 0, f"Number of calls during retime did not match original profile, originally {len(self._backend_calls)}, now {self._retime_index}, "
            self._retime_index = -1

    @contextmanager
    def _accumulate_average(self, n):
        self._retime_index = 0
        self._accumulating = True
        restore_data = _start_profiling(self, self._backends)
        try:
            yield None
        finally:
            _stop_profiling(self, *restore_data)
            assert self._retime_index == 0, f"Number of calls during retime did not match original profile, originally {len(self._backend_calls)}, now {self._retime_index}, "
            self._retime_index = -1
            for call in self._backend_calls:
                call._start /= n
                call._stop /= n
            self._accumulating = False


def _format_values(values: dict, backend):

    def format_val(value):
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, (int, float, complex, bool)):
            return value
        if isinstance(value, (tuple, list)):
            return str([format_val(v) for v in value])
        try:
            shape = backend.shape(value)
            dtype = backend.dtype(value)
            try:
                shape = (int(dim) if dim is not None else '?' for dim in shape)
            except Exception:
                pass
            return f"{tuple(shape)}, {dtype}"
        except BaseException:
            return str(value)

    lines = [f"{key}: {format_val(val)}" for key, val in values.items()]
    return "\n".join(lines)


class ProfilingBackend:

    def __init__(self, prof: Profile, backend: Backend, index: int):
        self._backend = backend
        self._profile = prof
        self._index = index
        # non-profiling methods
        self.name = backend.name
        self.combine_types = backend.combine_types
        self.auto_cast = backend.auto_cast
        self.is_tensor = backend.is_tensor
        self.is_available = backend.is_available
        self.shape = backend.shape
        self.staticshape = backend.staticshape
        self.ndims = backend.ndims
        self.dtype = backend.dtype
        self.expand_dims = backend.expand_dims
        self.reshape = backend.reshape
        self.supports = backend.supports
        # TODO strided slice does not go through backend atm
        # profiling methods
        for item_name in dir(backend):
            item = getattr(backend, item_name)
            if callable(item) and not hasattr(self, item_name):
                def context(item=item, item_name=item_name, profiling_backend=self):
                    def call_fun(*args, **kwargs):
                        start = perf_counter()
                        result = item(*args, **kwargs)
                        stop = perf_counter()
                        prof._add_call(BackendCall(start, stop, profiling_backend, item_name), args, kwargs, result)
                        return result
                    return call_fun
                setattr(self, item_name, context())

    def call(self, f: Callable, *args, name=None):
        start = perf_counter()
        result = f(*args)
        self._backend.block_until_ready(result)
        stop = perf_counter()
        self._profile._add_call(BackendCall(start, stop, self, name), args, {}, result)
        return result

    def __repr__(self):
        return f"profile[{self._backend}]"

    def __enter__(self):
        _DEFAULT.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _DEFAULT.pop(-1)

    def __eq__(self, other):
        return other is self or other is self._backend

    def __hash__(self):
        return hash(self._backend)


_PROFILE = []


@contextmanager
def profile(backends=None, trace=True, subtract_trace_time=True, save: str or None = None) -> Profile:
    """
    To be used in `with` statements, `with math.backend.profile() as prof: ...`.
    Creates a `Profile` for the code executed within the context by tracking calls to the `backends` and optionally tracing the call.

    Args:
        backends: List of backends to profile, `None` to profile all.
        trace: Whether to perform a full stack trace for each backend call. If true, groups backend calls by function.
        subtract_trace_time: If True, subtracts the time it took to trace the call stack from the event times
        save: (Optional) File path to save the profile to. This will call `Profile.save()`.

    Returns:
        Created `Profile`
    """
    backends = BACKENDS if backends is None else backends
    prof = Profile(trace, backends, subtract_trace_time)
    restore_data = _start_profiling(prof, backends)
    try:
        yield prof
    finally:
        _stop_profiling(prof, *restore_data)
        if save is not None:
            prof.save(save)


def profile_function(fun: Callable,
                     args: tuple or list = (),
                     kwargs: dict or None = None,
                     backends=None,
                     trace=True,
                     subtract_trace_time=True,
                     retime=True,
                     warmup=1,
                     call_count=1) -> Profile:
    """
    Creates a `Profile` for the function `fun(*args, **kwargs)`.

    Args:
        fun: Function to be profiled. In case `retime=True`, this function must perform the same operations each time it is called.
            Use `warmup>0` to ensure that internal caching does not interfere with the operations.
        args: Arguments to be passed to `fun`.
        kwargs: Keyword arguments to be passed to `fun`.
        backends: List of backends to profile, `None` to profile all.
        trace: Whether to perform a full stack trace for each backend call. If true, groups backend calls by function.
        subtract_trace_time: If True, subtracts the time it took to trace the call stack from the event times. Has no effect if `retime=True`.
        retime: If true, calls `fun` another time without tracing the calls and updates the profile.
            This gives a much better indication of the true timing.
            See `Profile.retime()`.
        warmup: Number of times to call `fun` before profiling it.
        call_count: How often to call the function (excluding retime and warmup). The times will be averaged over multiple runs if `call_count > 1`.

    Returns:
        Created `Profile` for `fun`.
    """
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    for _ in range(warmup):
        fun(*args, **kwargs)
    with profile(backends=backends, trace=trace, subtract_trace_time=subtract_trace_time) as prof:
        fun(*args, **kwargs)
    if retime:
        with prof.retime():
            fun(*args, **kwargs)
    if call_count > 1:
        with prof._accumulate_average(call_count):
            for _ in range(call_count - 1):
                fun(*args, **kwargs)
    return prof


def _start_profiling(prof: Profile, backends: tuple or list):
    _PROFILE.append(prof)
    original_default = _DEFAULT[-1]
    original_backends = tuple(BACKENDS)
    for i, backend in enumerate(backends):
        prof_backend = ProfilingBackend(prof, backend, i)
        BACKENDS[BACKENDS.index(backend)] = prof_backend
        if _DEFAULT[-1] == backend:
            _DEFAULT[-1] = prof_backend
    return original_backends, original_default


def _stop_profiling(prof: Profile, original_backends, original_default):
    prof._finish()
    _PROFILE.pop(-1)
    BACKENDS.clear()
    BACKENDS.extend(original_backends)
    _DEFAULT[-1] = original_default


def get_current_profile() -> Optional[Profile]:
    """ Returns the currently active `Profile` if one is active. Otherwise returns `None`.  """
    return _PROFILE[-1] if _PROFILE else None
