import functools
from copy import copy

import numpy
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import stax  # from jax.example_libraries import stax
from jax.experimental import optimizers as optim
from typing import Callable

from jax.experimental.optimizers import OptimizerState

from phi import math
from .. import JAX
from ...math._functional import JitFunction


class StaxNet:

    def __init__(self, initialize: Callable, apply: Callable, input_shape: tuple):
        self._initialize = initialize
        self._apply = apply
        self._input_shape = input_shape
        self.parameters = None
        self._tracers = None

    def initialize(self):
        rnd_key = JAX.rnd_key
        JAX.rnd_key, init_key = random.split(rnd_key)
        out_shape, params64 = self._initialize(init_key, input_shape=self._input_shape)
        if math.get_precision() < 64:
            self.parameters = _recursive_to_float32(params64)

    def __call__(self, *args, **kwargs):
        if self._tracers is not None:
            return self._apply(self._tracers, *args)
        else:
            return self._apply(self.parameters, *args)


class JaxOptimizer:

    def __init__(self, initialize: Callable, update: Callable, get_params: Callable):
        self._initialize, self._update, self._get_params = initialize, update, get_params  # Stax functions
        self._state = None
        self._step_i = 0
        self._update_function_cache = {}

    def initialize(self, net: tuple):
        self._state = self._initialize(net)

    def update_step(self, grads: tuple):
        self._state = self._update(self._step_i, grads, self._state)
        self._step_i += 1

    def get_network_parameters(self):
        return self._get_params(self._state)

    def update(self, net: StaxNet, loss_function, wrt, loss_args, loss_kwargs):
        if loss_function not in self._update_function_cache:
            def update(packed_current_state, *loss_args, **loss_kwargs):
                def loss_depending_on_net(params_tracer: tuple, *args, **kwargs):
                    net._tracers = params_tracer
                    loss_function_non_jit = loss_function.f if isinstance(loss_function, JitFunction) else loss_function
                    result = loss_function_non_jit(*args, **kwargs)
                    net._tracers = None
                    return result

                gradient_function = math.functional_gradient(loss_depending_on_net)
                current_state = OptimizerState(packed_current_state, self._state.tree_def, self._state.subtree_defs)
                current_params = self._get_params(current_state)
                value, grads = gradient_function(current_params, *loss_args, **loss_kwargs)
                next_state = self._update(self._step_i, grads[0], self._state)
                return next_state.packed_state, value

            if isinstance(loss_function, JitFunction):
                update = math.jit_compile(update)
            self._update_function_cache[loss_function] = update

        next_packed_state, loss_output = self._update_function_cache[loss_function](self._state.packed_state, *loss_args, **loss_kwargs)
        self._state = OptimizerState(next_packed_state, self._state.tree_def, self._state.subtree_defs)
        return loss_output



def parameter_count(model: StaxNet) -> int:
    """
    Counts the number of parameters in a model.

    Args:
        model: Stax model

    Returns:
        `int`
    """
    return int(_recursive_count_parameters(model.parameters))


def _recursive_to_float32(obj):
    if isinstance(obj, (tuple, list)):
        return type(obj)([_recursive_to_float32(i) for i in obj])
    elif isinstance(obj, dict):
        return {k: _recursive_to_float32(v) for k, v in obj.items()}
    else:
        assert isinstance(obj, jax.numpy.ndarray)
        return obj.astype(jax.numpy.float32)


def _recursive_count_parameters(obj):
    if isinstance(obj, (tuple, list)):
        return sum([_recursive_count_parameters(item) for item in obj])
    if isinstance(obj, dict):
        return sum([_recursive_count_parameters(v) for v in obj.values()])
    return numpy.prod(obj.shape)


def save_state(obj: StaxNet or JaxOptimizer, path: str):
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.npy'):
        path += '.npy'
    if isinstance(obj, StaxNet):
        numpy.save(path, obj.parameters)
    else:
        pass  # ToDo
        # numpy.save(path, obj._state)


def load_state(obj: StaxNet or JaxOptimizer, path: str):
    """
    Read the state of a module or optimizer from a file.

    See Also:
        `save_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.npy'):
        path += '.npy'
    if isinstance(obj, StaxNet):
        state = numpy.load(path, allow_pickle=True)
        obj.parameters = tuple([tuple(layer) for layer in state])
    else:
        pass  # ToDo


def update_weights(net: StaxNet, optimizer: JaxOptimizer, loss_function: Callable, *loss_args, **loss_kwargs):
    """
    Computes the gradients of `loss_function` w.r.t. the parameters of `net` and updates its weights using `optimizer`.

    This is the Jax version. Analogue functions exist for other learning frameworks.

    Args:
        net: Learning model.
        optimizer: Optimizer.
        loss_function: Loss function, called as `loss_function(*loss_args, **loss_kwargs)`.
        *loss_args: Arguments given to `loss_function`.
        **loss_kwargs: Keyword arguments given to `loss_function`.

    Returns:
        Output of `loss_function`.
    """
    loss_output = optimizer.update(net, loss_function, net.parameters, loss_args, loss_kwargs)
    net.parameters = optimizer.get_network_parameters()
    return loss_output


def adam(net: StaxNet, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    """
    Creates an Adam optimizer for `net`, alias for [`jax.example_libraries.optimizers.adam`](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html).
    Analogue functions exist for other learning frameworks.
    """
    opt = JaxOptimizer(*optim.adam(learning_rate, betas[0], betas[1], epsilon))
    opt.initialize(net.parameters)
    return opt


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              batch_norm=False,
              activation='ReLU') -> StaxNet:
    activation = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh}[activation]
    stax_layers = []
    for neuron_count in layers:
        stax_layers.append(stax.Dense(neuron_count))
        stax_layers.append(activation)
        if batch_norm:
            stax_layers.append(stax.BatchNorm(axis=(0,)))
    stax_layers.append(stax.Dense(out_channels))
    net_init, net_apply = stax.serial(*stax_layers)
    net = StaxNet(net_init, net_apply, (-1, in_channels))
    net.initialize()
    return net


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm: bool = True,
          activation='ReLU',
          in_spatial: tuple or int = 2) -> StaxNet:
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation]
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (-1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    # Create layers
    inc_init, inc_apply = create_double_conv(d, filters[0], filters[0], batch_norm, activation)
    init_functions, apply_functions = {}, {}
    for i in range(1, levels):
        init_functions[f'down{i}'], apply_functions[f'down{i}'] = create_double_conv(d, filters[i], filters[i], batch_norm, activation)
        init_functions[f'up{i}'], apply_functions[f'up{i}'] = create_double_conv(d, filters[i - 1], filters[i - 1], batch_norm, activation)
    outc_init, outc_apply = CONV[d](out_channels, (1,) * d, padding='same')
    max_pool_init, max_pool_apply = stax.MaxPool((2,) * d, padding='same', strides=(2,) * d)
    _, up_apply = create_upsample()

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)
        shape = input_shape
        # Layers
        shape, params['inc'] = inc_init(rngs[0], shape)
        shapes = [shape]
        for i in range(1, levels):
            shape, _ = max_pool_init(None, shape)
            shape, params[f'down{i}'] = init_functions[f'down{i}'](rngs[i], shape)
            shapes.insert(0, shape)
        for i in range(1, levels):
            shape = shapes[i][:-1] + (shapes[i][-1] + shape[-1],)
            shape, params[f'up{i}'] = init_functions[f'up{i}'](rngs[levels+i], shape)
        shape, params['outc'] = outc_init(rngs[-1], shape)
        return shape, params

    # no @jax.jit needed here since the user can jit this in the loss_function
    def net_apply(params, inputs, **kwargs):
        x = inputs
        x = inc_apply(params['inc'], x, **kwargs)
        xs = [x]
        for i in range(1, levels):
            x = max_pool_apply(None, x, **kwargs)
            x = apply_functions[f'down{i}'](params[f'down{i}'], x, **kwargs)
            xs.insert(0, x)
        for i in range(1, levels):
            x = up_apply(None, x, **kwargs)
            x = jnp.concatenate([x, xs[i]], axis=-1)
            x = apply_functions[f'up{i}'](params[f'up{i}'], x, **kwargs)
        x = outc_apply(params['outc'], x, **kwargs)
        return x

    net = StaxNet(net_init, net_apply, (-1,) + in_spatial + (in_channels,))
    net.initialize()
    return net


ACTIVATIONS = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh}
CONV = [None,
        functools.partial(stax.GeneralConv, ('NWC', 'WIO', 'NWC')),
        functools.partial(stax.GeneralConv, ('NWHC', 'WHIO', 'NWHC')),
        functools.partial(stax.GeneralConv, ('NWHDC', 'WHDIO', 'NWHDC')),
]


def create_double_conv(d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable):
    return stax.serial(
        CONV[d](mid_channels, (3,) * d, padding='same'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation,
        CONV[d](out_channels, (3,) * d, padding='same'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation,
    )


def create_upsample():
    # def upsample_init(rng, input_shape):
    #     return shape, []
    def upsample_apply(params, inputs, **kwargs):
        x = math.wrap(inputs, math.batch('batch'), *[math.spatial(f'{i}') for i in range(len(inputs.shape) - 2)], math.channel('vector'))
        x = math.upsample2x(x)
        return x.native(x.shape)
    return NotImplemented, upsample_apply
