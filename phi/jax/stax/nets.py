"""
Stax implementation of the unified machine learning API.
Equivalent functions also exist for the other frameworks.

For API documentation, see https://tum-pbs.github.io/PhiFlow/Network_API .
"""
import functools
import inspect
import warnings
from typing import Callable, Tuple, List

import numpy
import jax
import jax.numpy as jnp
from jax import random

from packaging import version

if version.parse(jax.__version__) >= version.parse(
        '0.2.25'):  # Stax and Optimizers were moved to jax.example_libraries on Oct 20, 2021
    from jax.example_libraries import stax
    import jax.example_libraries.optimizers as optim
    from jax.example_libraries.optimizers import OptimizerState
else:
    from jax.experimental import stax
    import jax.experimental.optimizers as optim
    from jax.experimental.optimizers import OptimizerState

    warnings.warn(f"Found Jax version {jax.__version__}. Using legacy imports.", FutureWarning)

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
            @functools.wraps(loss_function)
            def update(packed_current_state, *loss_args, **loss_kwargs):
                @functools.wraps(loss_function)
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

        next_packed_state, loss_output = self._update_function_cache[loss_function](self._state.packed_state,
                                                                                    *loss_args, **loss_kwargs)
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


def get_parameters(model: StaxNet, wrap=True) -> dict:
    result = {}
    _recursive_add_parameters(model.parameters, wrap, (), result)
    return result


def _recursive_add_parameters(param, wrap: bool, prefix: tuple, result: dict):
    if isinstance(param, dict):
        for name, obj in param.items():
            _recursive_add_parameters(obj, wrap, prefix + (name,), result)
    elif isinstance(param, (tuple, list)):
        for i, obj in enumerate(param):
            _recursive_add_parameters(obj, wrap, prefix + (i,), result)
    else:
        rank = len(param.shape)
        if prefix[-1] == 0 and rank == 2:
            name = '.'.join(str(p) for p in prefix[:-1]) + '.weight'
        elif prefix[-1] == 1 and rank == 1:
            name = '.'.join(str(p) for p in prefix[:-1]) + '.bias'
        else:
            name = '.'.join(prefix)
        if not wrap:
            result[name] = param
        else:
            if rank == 1:
                phi_tensor = math.wrap(param, math.channel('output'))
            elif rank == 2:
                phi_tensor = math.wrap(param, math.channel('input,output'))
            else:
                raise NotImplementedError
            result[name] = phi_tensor


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
        raise NotImplementedError  # ToDo
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
        raise NotImplementedError  # ToDo


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
    Analogous functions exist for other learning frameworks.
    """
    opt = JaxOptimizer(*optim.adam(learning_rate, betas[0], betas[1], epsilon))
    opt.initialize(net.parameters)
    return opt


def sgd(net: StaxNet, learning_rate: float = 1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    """
    Creates an SGD optimizer for `net`, alias for [`jax.example_libraries.optimizers.SGD`](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html).
    Analogous functions exist for other learning frameworks.
    """
    if momentum == 0:
        opt = JaxOptimizer(*optim.sgd(learning_rate))
    else:
        opt = JaxOptimizer(*optim.momentum(learning_rate, momentum))
    opt.initialize(net.parameters)
    return opt


def adagrad(net: StaxNet, learning_rate: float = 1e-3, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
    """
    Creates an Adagrad optimizer for `net`, alias for [`jax.example_libraries.optimizers.adagrad`](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html).
    Analogue functions exist for other learning frameworks.
    """
    opt = JaxOptimizer(*optim.adagrad(learning_rate))
    opt.initialize(net.parameters)
    return opt


def rmsprop(net: StaxNet, learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
    """
    Creates an RMSprop optimizer for `net`, alias for [`jax.example_libraries.optimizers.rmsprop`](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html).
    Analogue functions exist for other learning frameworks.
    """
    if momentum == 0:
        opt = JaxOptimizer(*optim.rmsprop(learning_rate, alpha, eps))
    else:
        opt = JaxOptimizer(*optim.rmsprop_momentum(learning_rate, alpha, eps, momentum))
    opt.initialize(net.parameters)
    return opt


def dense_net(in_channels: int,
              out_channels: int,
              layers: Tuple[int, ...] or List[int],
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
          in_spatial: tuple or int = 2,
          use_res_blocks: bool = False) -> StaxNet:
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation]
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    # Create layers
    if use_res_blocks:
        inc_init, inc_apply = resnet_block(in_channels, filters[0], batch_norm, activation, d)
    else:
        inc_init, inc_apply = create_double_conv(d, filters[0], filters[0], batch_norm, activation)
    init_functions, apply_functions = {}, {}
    for i in range(1, levels):
        if use_res_blocks:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = resnet_block(filters[i - 1], filters[i],
                                                                                   batch_norm, activation, d)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = resnet_block(filters[i] + filters[i - 1],
                                                                               filters[i - 1], batch_norm, activation,
                                                                               d)
        else:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = create_double_conv(d, filters[i], filters[i],
                                                                                         batch_norm, activation)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = create_double_conv(d, filters[i - 1], filters[i - 1],
                                                                                     batch_norm, activation)
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
            shape, params[f'up{i}'] = init_functions[f'up{i}'](rngs[levels + i], shape)
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

    net = StaxNet(net_init, net_apply, (1,) + in_spatial + (in_channels,))
    net.initialize()
    return net


ACTIVATIONS = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh, 'SiLU': stax.Selu}
CONV = [None,
        functools.partial(stax.GeneralConv, ('NWC', 'WIO', 'NWC')),
        functools.partial(stax.GeneralConv, ('NWHC', 'WHIO', 'NWHC')),
        functools.partial(stax.GeneralConv, ('NWHDC', 'WHDIO', 'NWHDC')), ]

'''
def create_double_conv(d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable):
    
    return stax.serial(
        CONV[d](out_channels, (3,) * d, padding='same'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation,
        CONV[d](out_channels, (3,) * d, padding='same'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation)
'''


# Periodic Implementation
def create_double_conv(d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable):
    init_fn, apply_fn = {}, {}

    init_fn['conv1'], apply_fn['conv1'] = stax.serial(CONV[d](mid_channels, (3,) * d, padding='valid'),
                                                      stax.BatchNorm(
                                                          axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
                                                      activation)

    init_fn['conv2'], apply_fn['conv2'] = stax.serial(CONV[d](mid_channels, (3,) * d, padding='valid'),
                                                      stax.BatchNorm(
                                                          axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
                                                      activation)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        shape, params['conv1'] = init_fn['conv1'](rngs[0], input_shape)
        shape, params['conv2'] = init_fn['conv2'](rngs[1], shape)

        return shape, params

    def net_apply(params, inputs):
        x = inputs

        pad_tuple = [[0, 0]] + [[1, 1] for i in range(d)] + [[0, 0]]

        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap')
        out = apply_fn['conv1'](params['conv1'], out)
        out = jnp.pad(out, pad_width=pad_tuple, mode='wrap')
        out = apply_fn['conv2'](params['conv2'], out)

        return out

    return net_init, net_apply


def create_upsample():
    # def upsample_init(rng, input_shape):
    #     return shape, []
    def upsample_apply(params, inputs, **kwargs):
        x = math.wrap(inputs, math.batch('batch'), *[math.spatial(f'{i}') for i in range(len(inputs.shape) - 2)],
                      math.channel('vector'))
        x = math.upsample2x(x)
        return x.native(x.shape)

    return NotImplemented, upsample_apply


def conv_classifier(input_shape_list: list, num_classes: int, batch_norm: bool, in_spatial: int or tuple):
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    stax_conv_layers = []
    stax_dense_layers = []
    spatial_shape_list = list(input_shape_list[1:])

    in_channels = input_shape_list[0]

    channels = [64, 128, 256, 512, 512]

    init_fn, apply_fn = {}, {}
    init_fn['conv1'], apply_fn['conv1'] = create_double_conv(d, 64, 64, batch_norm, ACTIVATIONS['ReLU'])
    init_fn['max_pool1'], apply_fn['max_pool1'] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)

    init_fn['conv2'], apply_fn['conv2'] = create_double_conv(d, 128, 128, batch_norm, ACTIVATIONS['ReLU'])
    init_fn['max_pool2'], apply_fn['max_pool2'] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)

    init_fn['conv3_1'], apply_fn['conv3_1'] = create_double_conv(d, 256, 256, batch_norm, ACTIVATIONS['ReLU'])
    init_fn['conv3_2'], apply_fn['conv3_2'] = stax.serial(CONV[d](256, (3,) * d, padding='valid'),
                                                          stax.BatchNorm(axis=tuple(
                                                              range(d + 1))) if batch_norm else stax.Identity,
                                                          ACTIVATIONS['ReLU'])

    init_fn['max_pool3'], apply_fn['max_pool3'] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)

    init_fn['conv4_1'], apply_fn['conv4_1'] = create_double_conv(d, 512, 512, batch_norm, ACTIVATIONS['ReLU'])
    init_fn['conv4_2'], apply_fn['conv4_2'] = stax.serial(CONV[d](512, (3,) * d, padding='valid'),
                                                          stax.BatchNorm(axis=tuple(
                                                              range(d + 1))) if batch_norm else stax.Identity,
                                                          ACTIVATIONS['ReLU'])
    init_fn['max_pool4'], apply_fn['max_pool4'] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)

    init_fn['conv5_1'], apply_fn['conv5_1'] = create_double_conv(d, 512, 512, batch_norm, ACTIVATIONS['ReLU'])
    init_fn['conv5_2'], apply_fn['conv5_2'] = stax.serial(CONV[d](512, (3,) * d, padding='valid'),
                                                          stax.BatchNorm(axis=tuple(
                                                              range(d + 1))) if batch_norm else stax.Identity,
                                                          ACTIVATIONS['ReLU'])
    init_fn['max_pool5'], apply_fn['max_pool5'] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)

    net_list = ['conv1', 'max_pool1', 'conv2', 'max_pool2',
                'conv3_1', 'conv3_2', 'max_pool3',
                'conv4_1', 'conv4_2', 'max_pool4',
                'conv5_1', 'conv5_2', 'max_pool5']
    init_fn['flatten'], apply_fn['flatten'] = stax.Flatten

    dense_layers = [4096, 4096, 100]
    for i, neuron_count in enumerate(dense_layers):
        stax_dense_layers.append(stax.Dense(neuron_count))
        stax_dense_layers.append(ACTIVATIONS['ReLU'])
        if batch_norm:
            stax_dense_layers.append(stax.BatchNorm(axis=(0,)))
    stax_dense_layers.append(stax.Dense(num_classes))
    softmax = stax.elementwise(stax.softmax, axis=-1)

    stax_dense_layers.append(softmax)

    dense_init, dense_apply = stax.serial(*stax_dense_layers)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        for i in range(5):
            for j in range(len(spatial_shape_list)):
                spatial_shape_list[j] = math.floor((spatial_shape_list[j] - 2) / 2) + 1

        flattened_input_dim = 1
        for i in range(len(spatial_shape_list)):
            flattened_input_dim *= spatial_shape_list[i]
        flattened_input_dim *= 512
        flattened_input_dim = int(flattened_input_dim)

        shape = input_shape

        N = len(net_list)
        for i in range(N):
            shape, params[f'{net_list[i]}'] = \
                init_fn[f'{net_list[i]}'](rngs[i], shape)

        shape, params['flatten'] = init_fn['flatten'](rngs[N], shape)
        shape, params['dense'] = dense_init(rngs[N + 1], (1,) + (flattened_input_dim,))

        return shape, params

    def net_apply(params, inputs, **kwargs):
        x = inputs

        pad_tuple = [[0, 0]] + [[1, 1] for i in range(d)] + [[0, 0]]

        for i in range(len(net_list)):
            if net_list[i] in ['conv3_2', 'conv4_2', 'conv5_2']:
                x = jnp.pad(x, pad_width=pad_tuple, mode='wrap')
            x = apply_fn[f'{net_list[i]}'](params[f'{net_list[i]}'], x)

        x = apply_fn['flatten'](params['flatten'], x)
        out = dense_apply(params['dense'], x, **kwargs)
        return out

    net = StaxNet(net_init, net_apply, (1,) + in_spatial + (in_channels,))
    net.initialize()
    return net


def conv_net(in_channels: int,
             out_channels: int,
             layers: Tuple[int, ...] or List[int],
             batch_norm: bool = False,
             activation: str or Callable = 'ReLU',
             in_spatial: int or tuple = 2) -> StaxNet:
    if isinstance(in_spatial, tuple):
        d = in_spatial
        in_spatial = len(in_spatial)
    else:
        d = (1,) * in_spatial
    if isinstance(activation, str):
        activation = ACTIVATIONS[activation]

    stax_layers = []

    init_fn, apply_fn = {}, {}
    for i in range(len(layers)):
        init_fn[f'conv{i + 1}'], apply_fn[f'conv{i + 1}'] = stax.serial(
            CONV[in_spatial](out_channels, (3,) * in_spatial, padding='valid'),
            stax.BatchNorm(axis=tuple(range(in_spatial + 1))) if batch_norm else stax.Identity,
            activation)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        shape, params['conv1'] = init_fn['conv1'](rngs[0], input_shape)
        for i in range(1, len(layers)):
            shape, params[f'conv{i + 1}'] = init_fn[f'conv{i + 1}'](rngs[i], shape)

        return shape, params

    def net_apply(params, inputs):
        x = inputs

        pad_tuple = [(0, 0)]
        for i in range(in_spatial):
            pad_tuple.append((1, 1))
        pad_tuple.append((0, 0))

        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap')
        out = apply_fn['conv1'](params['conv1'], out)
        for i in range(1, len(layers)):
            out = jnp.pad(out, pad_width=pad_tuple, mode='wrap')
            out = apply_fn[f'conv{i + 1}'](params[f'conv{i + 1}'], out)
        return out

    net = StaxNet(net_init, net_apply, (1,) + d + (in_channels,))
    net.initialize()
    return net


def res_net(in_channels: int,
            out_channels: int,
            layers: Tuple[int, ...] or List[int],
            batch_norm: bool = False,
            activation: str or Callable = 'ReLU',
            in_spatial: int or tuple = 2) -> StaxNet:
    if isinstance(in_spatial, tuple):
        d = in_spatial
        in_spatial = len(in_spatial)
    else:
        d = (1,) * in_spatial

    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    stax_layers = []

    stax_layers.append(resnet_block(in_channels, layers[0], batch_norm, activation, in_spatial))

    for i in range(1, len(layers)):
        stax_layers.append(resnet_block(layers[i - 1], layers[i], batch_norm, activation, in_spatial))

    stax_layers.append(resnet_block(layers[len(layers) - 1], out_channels, batch_norm, activation, in_spatial))
    net_init, net_apply = stax.serial(*stax_layers)
    net = StaxNet(net_init, net_apply, (1,) + d + (in_channels,))
    net.initialize()
    return net


def resnet_block(in_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 activation: str or Callable = 'ReLU',
                 in_spatial: int or tuple = 2):
    if isinstance(in_spatial, int):
        d = (1,) * in_spatial
    else:
        d = in_spatial
        in_spatial = len(in_spatial)

    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    init_fn, apply_fn = {}, {}
    init_fn['conv1'], apply_fn['conv1'] = stax.serial(
        CONV[in_spatial](out_channels, (3,) * in_spatial, padding='valid'),
        stax.BatchNorm(axis=tuple(range(in_spatial + 1))) if batch_norm else stax.Identity,
        activation)
    init_fn['conv2'], apply_fn['conv2'] = stax.serial(
        CONV[in_spatial](out_channels, (3,) * in_spatial, padding='valid'),
        stax.BatchNorm(axis=tuple(range(in_spatial + 1))) if batch_norm else stax.Identity,
        activation)

    init_activation, apply_activation = activation
    if in_channels != out_channels:
        init_fn['sample_conv'], apply_fn['sample_conv'] = stax.serial(
            CONV[in_spatial](out_channels, (1,) * in_spatial, padding='VALID'),
            stax.BatchNorm(axis=tuple(range(in_spatial + 1))) if batch_norm else stax.Identity)
    else:
        init_fn['sample_conv'], apply_fn['sample_conv'] = stax.Identity

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        # Preparing a list of shapes and dictionary of parameters to return
        shape, params['conv1'] = init_fn['conv1'](rngs[0], input_shape)
        shape, params['conv2'] = init_fn['conv2'](rngs[1], shape)
        shape, params['sample_conv'] = init_fn['sample_conv'](rngs[2], input_shape)
        shape, params['activation'] = init_activation(rngs[3], shape)
        return shape, params

    def net_apply(params, inputs, **kwargs):
        x = inputs

        pad_tuple = [[0, 0]] + [[1, 1] for i in range(in_spatial)] + [[0, 0]]

        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap')
        out = apply_fn['conv1'](params['conv1'], out)
        out = jnp.pad(out, pad_width=pad_tuple, mode='wrap')
        out = apply_fn['conv2'](params['conv2'], out)
        skip_x = apply_fn['sample_conv'](params['sample_conv'], x, **kwargs)
        out = jnp.add(out, skip_x)
        # out = apply_activation(params['activation'], out)
        return out

    return net_init, net_apply
