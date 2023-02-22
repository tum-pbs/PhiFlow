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
import keras
import numpy
import numpy as np
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
            _recursive_add_parameters(obj, wrap, prefix + (str(name),), result)
    elif isinstance(param, (tuple, list)):
        for i, obj in enumerate(param):
            _recursive_add_parameters(obj, wrap, prefix + (str(i),), result)
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
            elif rank == 3:
                phi_tensor = math.wrap(param, math.channel('x,input,output'))
            elif rank == 4:
                phi_tensor = math.wrap(param, math.channel('x,y,input,output'))
            elif rank == 5:
                phi_tensor = math.wrap(param, math.channel('x,y,z,input,output'))
            else:
                raise NotImplementedError(rank)
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
              activation='ReLU',
              softmax=False) -> StaxNet:
    """
    Fully-connected neural networks are available in ΦFlow via dense_net().
    Arguments:
        in_channels : size of input layer, int
        out_channels = size of output layer, int
        layers : tuple of linear layers between input and output neurons, list or tuple
        activation : activation function used within the layers, string
        batch_norm : use of batch norm after each linear layer, bool

    Returns:
        Dense net model as specified by input arguments
    """
    activation = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh}[activation]
    stax_layers = []
    for neuron_count in layers:
        stax_layers.append(stax.Dense(neuron_count))
        stax_layers.append(activation)
        if batch_norm:
            stax_layers.append(stax.BatchNorm(axis=(0,)))
    stax_layers.append(stax.Dense(out_channels))
    if softmax:
        stax_layers.append(stax.elementwise(stax.softmax, axis=-1))
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
          periodic=False,
          use_res_blocks: bool = False) -> StaxNet:
    """
     ΦFlow provides a built-in U-net architecture, classically popular for Semantic Segmentation in Computer Vision, composed of downsampling and upsampling layers.

     Arguments:

         in_channels: input channels of the feature map, dtype : int
         out_channels : output channels of the feature map, dtype : int
         levels : number of levels of down-sampling and upsampling, dtype : int
         filters : filter sizes at each down/up sampling convolutional layer, if the input is integer all conv layers have the same filter size,
         dtype : int or tuple
         activation : activation function used within the layers, dtype : string
         batch_norm : use of batchnorm after each conv layer, dtype : bool
         in_spatial : spatial dimensions of the input feature map, dtype : int
         use_res_blocks : use convolutional blocks with skip connections instead of regular convolutional blocks, dtype : bool

     Returns:

         U-net model as specified by input arguments
     """
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    # Create layers
    if use_res_blocks:
        inc_init, inc_apply = resnet_block(in_channels, filters[0], periodic, batch_norm, activation, d)
    else:
        inc_init, inc_apply = create_double_conv(d, filters[0], filters[0], batch_norm, activation, periodic)
    init_functions, apply_functions = {}, {}
    for i in range(1, levels):
        if use_res_blocks:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = resnet_block(filters[i - 1], filters[i], periodic, batch_norm, activation, d)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = resnet_block(filters[i] + filters[i - 1], filters[i - 1], periodic, batch_norm, activation, d)
        else:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = create_double_conv(d, filters[i], filters[i], batch_norm, activation, periodic)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = create_double_conv(d, filters[i - 1], filters[i - 1], batch_norm, activation, periodic)
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
def create_double_conv(d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable, periodic: bool):
    init_fn, apply_fn = {}, {}
    init_fn['conv1'], apply_fn['conv1'] = stax.serial(CONV[d](mid_channels, (3,) * d, padding='valid'), stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity, activation)
    init_fn['conv2'], apply_fn['conv2'] = stax.serial(CONV[d](mid_channels, (3,) * d, padding='valid'), stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity, activation)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)
        shape, params['conv1'] = init_fn['conv1'](rngs[0], input_shape)
        shape, params['conv2'] = init_fn['conv2'](rngs[1], shape)

        return shape, params

    def net_apply(params, inputs):
        x = inputs
        pad_tuple = [[0, 0]] + [[1, 1] for i in range(d)] + [[0, 0]]
        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
        out = apply_fn['conv1'](params['conv1'], out)
        out = jnp.pad(out, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
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


def conv_classifier(in_features: int,
                    in_spatial: tuple or list,
                    num_classes: int,
                    blocks=(64, 128, 256, 256, 512, 512),
                    dense_layers=(4096, 4096, 100),
                    batch_norm=True,
                    activation='ReLU',
                    softmax=True,
                    periodic=False):
    """
    Based on VGG16.
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    stax_dense_layers = []
    init_fn, apply_fn = {}, {}

    net_list = []
    for i, (prev, next) in enumerate(zip((in_features,) + tuple(blocks[:-1]), blocks)):
        if i in (0, 1):
            net_list.append(f'conv{i+1}')
            init_fn[net_list[-1]], apply_fn[net_list[-1]] = create_double_conv(d, next, next, batch_norm, activation, periodic)
        else:
            net_list.append(f'conv{i+1}_1')
            init_fn[net_list[-1]], apply_fn[net_list[-1]] = create_double_conv(d, 256, 256, batch_norm, activation, periodic)
            net_list.append(f'conv{i+1}_2')
            init_fn[net_list[-1]], apply_fn[net_list[-1]] = stax.serial(CONV[d](256, (3,) * d, padding='valid'),
                                                                        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
                                                                        activation)
        net_list.append(f'max_pool{i+1}')
        init_fn[net_list[-1]], apply_fn[net_list[-1]] = stax.MaxPool((2,) * d, padding='valid', strides=(2,) * d)
    init_fn['flatten'], apply_fn['flatten'] = stax.Flatten
    for i, neuron_count in enumerate(dense_layers):
        stax_dense_layers.append(stax.Dense(neuron_count))
        stax_dense_layers.append(activation)
        if batch_norm:
            stax_dense_layers.append(stax.BatchNorm(axis=(0,)))
    stax_dense_layers.append(stax.Dense(num_classes))
    if softmax:
        stax_dense_layers.append(stax.elementwise(stax.softmax, axis=-1))
    dense_init, dense_apply = stax.serial(*stax_dense_layers)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)
        shape = input_shape
        N = len(net_list)
        for i in range(N):
            shape, params[f'{net_list[i]}'] = init_fn[f'{net_list[i]}'](rngs[i], shape)
        shape, params['flatten'] = init_fn['flatten'](rngs[N], shape)
        flat_size = int(np.prod(in_spatial) * blocks[-1] / (2**d) ** len(blocks))
        shape, params['dense'] = dense_init(rngs[N + 1], (1,) + (flat_size,))
        return shape, params

    def net_apply(params, inputs, **kwargs):
        x = inputs
        pad_tuple = [[0, 0]] + [[1, 1]] * d + [[0, 0]]
        for i in range(len(net_list)):
            if net_list[i] in ['conv3_2', 'conv4_2', 'conv5_2']:
                x = jnp.pad(x, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
            x = apply_fn[f'{net_list[i]}'](params[f'{net_list[i]}'], x)
        x = apply_fn['flatten'](params['flatten'], x)
        out = dense_apply(params['dense'], x, **kwargs)
        return out

    net = StaxNet(net_init, net_apply, (1,) + in_spatial + (in_features,))
    net.initialize()
    return net


def conv_net(in_channels: int,
             out_channels: int,
             layers: Tuple[int, ...] or List[int],
             batch_norm: bool = False,
             activation: str or Callable = 'ReLU',
             periodic=False,
             in_spatial: int or tuple = 2) -> StaxNet:
    """
    Built in Conv-Nets are also provided. Contrary to the classical convolutional neural networks, the feature map spatial size remains the same throughout the layers. Each layer of the network is essentially a convolutional block comprising of two conv layers. A filter size of 3 is used in the convolutional layers.
    Arguments:

        in_channels : input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        layers : list or tuple of output channels for each intermediate layer between the input and final output channels, dtype : list or tuple
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int

    Returns:

        Conv-net model as specified by input arguments
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    init_fn, apply_fn = {}, {}
    if len(layers) < 1:
        layers.append(out_channels)
    init_fn['conv_in'], apply_fn['conv_in'] = stax.serial(
        CONV[d](layers[0], (3,) * d, padding='valid'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation)
    for i in range(1, len(layers)):
        init_fn[f'conv{i}'], apply_fn[f'conv{i}'] = stax.serial(
            CONV[d](layers[i], (3,) * d, padding='valid'),
            stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
            activation)
    init_fn['conv_out'], apply_fn['conv_out'] = CONV[d](out_channels, (1,) * d)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)
        shape, params['conv_in'] = init_fn['conv_in'](rngs[0], input_shape)
        for i in range(1, len(layers)):
            shape, params[f'conv{i + 1}'] = init_fn[f'conv{i + 1}'](rngs[i], shape)
        shape, params['conv_out'] = init_fn['conv_out'](rngs[len(layers)], shape)
        return shape, params

    def net_apply(params, inputs):
        x = inputs
        pad_tuple = [(0, 0)]
        for i in range(d):
            pad_tuple.append((1, 1))
        pad_tuple.append((0, 0))
        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
        out = apply_fn['conv_in'](params['conv_in'], out)
        for i in range(1, len(layers)):
            out = jnp.pad(out, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
            out = apply_fn[f'conv{i + 1}'](params[f'conv{i + 1}'], out)
        out = apply_fn['conv_out'](params['conv_out'], out)
        return out

    net = StaxNet(net_init, net_apply, (1,) + in_spatial + (in_channels,))
    net.initialize()
    return net


def res_net(in_channels: int,
            out_channels: int,
            layers: Tuple[int, ...] or List[int],
            batch_norm: bool = False,
            activation: str or Callable = 'ReLU',
            periodic=False,
            in_spatial: int or tuple = 2) -> StaxNet:
    """
    Built in Res-Nets are provided in the ΦFlow framework. Similar to the conv-net, the feature map spatial size remains the same throughout the layers.
    These networks use residual blocks composed of two conv layers with a skip connection added from the input to the output feature map.
    A default filter size of 3 is used in the convolutional layers.

    Arguments:

        in_channels : input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        layers : list or tuple of output channels for each intermediate layer between the input and final output channels, dtype : list or tuple
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int

    Returns:

        Res-net model as specified by input arguments
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    stax_layers = []
    if len(layers) > 0:
        stax_layers.append(resnet_block(in_channels, layers[0], periodic, batch_norm, activation, d))

        for i in range(1, len(layers)):
            stax_layers.append(resnet_block(layers[i - 1], layers[i], periodic, batch_norm, activation, d))

        stax_layers.append(resnet_block(layers[len(layers) - 1], out_channels, periodic, batch_norm, activation, d))
    else:
        stax_layers.append(resnet_block(in_channels, out_channels, periodic, batch_norm, activation, d))
    net_init, net_apply = stax.serial(*stax_layers)
    net = StaxNet(net_init, net_apply, (1,) + in_spatial + (in_channels,))
    net.initialize()
    return net


def resnet_block(in_channels: int,
                 out_channels: int,
                 periodic: bool,
                 batch_norm: bool,
                 activation: str or Callable = 'ReLU',
                 d: int or tuple = 2):
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    init_fn, apply_fn = {}, {}
    init_fn['conv1'], apply_fn['conv1'] = stax.serial(
        CONV[d](out_channels, (3,) * d, padding='valid'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation)
    init_fn['conv2'], apply_fn['conv2'] = stax.serial(
        CONV[d](out_channels, (3,) * d, padding='valid'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation)

    init_activation, apply_activation = activation
    if in_channels != out_channels:
        init_fn['sample_conv'], apply_fn['sample_conv'] = stax.serial(
            CONV[d](out_channels, (1,) * d, padding='VALID'),
            stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity)
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

        pad_tuple = [[0, 0]] + [[1, 1] for i in range(d)] + [[0, 0]]

        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
        out = apply_fn['conv1'](params['conv1'], out)
        out = jnp.pad(out, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
        out = apply_fn['conv2'](params['conv2'], out)
        skip_x = apply_fn['sample_conv'](params['sample_conv'], x, **kwargs)
        out = jnp.add(out, skip_x)
        # out = apply_activation(params['activation'], out)
        return out

    return net_init, net_apply


def get_mask(inputs, reverse_mask, data_format='NHWC'):
    """ Compute mask for slicing input feature map for Invertible Nets """
    shape = inputs.shape
    if len(shape) == 2:
        N = shape[-1]
        range_n = jnp.arange(0, N)
        even_ind = range_n % 2
        checker = jnp.reshape(even_ind, (-1, N))
    elif len(shape) == 4:
        H = shape[2] if data_format == 'NCHW' else shape[1]
        W = shape[3] if data_format == 'NCHW' else shape[2]

        range_h = jnp.arange(0, H) % 2
        range_w = jnp.arange(0, W) % 2

        even_ind_h = range_h.astype(bool)
        even_ind_w = range_w.astype(bool)

        ind_h = jnp.tile(jnp.expand_dims(even_ind_h, -1), [1, W])
        ind_w = jnp.tile(jnp.expand_dims(even_ind_w, 0), [H, 1])
        # ind_h = even_ind_h.unsqueeze(-1).repeat(1, W)
        # ind_w = even_ind_w.unsqueeze( 0).repeat(H, 1)

        checker = jnp.logical_xor(ind_h, ind_w)

        reshape = [-1, 1, H, W] if data_format == 'NCHW' else [-1, H, W, 1]
        checker = jnp.reshape(checker, reshape)
        checker = checker.astype(jnp.float32)

    else:
        raise ValueError('Invalid tensor shape. Dimension of the tensor shape must be '
                         '2 (NxD) or 4 (NxCxHxW or NxHxWxC), got {}.'.format(inputs.get_shape().as_list()))

    if reverse_mask:
        checker = 1 - checker

    return checker


def Dense_resnet_block(in_channels: int,
                       mid_channels: int,
                       batch_norm: bool = False,
                       activation: str or Callable = 'ReLU'):
    inputs = keras.Input(shape=(in_channels,))
    x_1 = inputs

    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    init_fn, apply_fn = {}, {}
    init_fn['dense1'], apply_fn['dense1'] = stax.serial(stax.Dense(mid_channels),
                                                        stax.BatchNorm(axis=(0,)),
                                                        activation)
    init_fn['dense2'], apply_fn['dense2'] = stax.serial(stax.Dense(in_channels),
                                                        stax.BatchNorm(axis=(0,)),
                                                        activation)
    init_activation, apply_activation = activation

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        shape, params['dense1'] = init_fn['dense1'](rngs[0], input_shape)
        shape, params['dense2'] = init_fn['dense2'](rngs[1], shape)
        shape, params['activation'] = init_activation(rngs[2], shape)
        return shape, params

    def net_apply(params, inputs, **kwargs):
        x = inputs

        out = apply_fn['dense1'](params['dense1'], x)
        out = apply_fn['dense2'](params['dense2'], out)

        out = jnp.add(out, x)

        return out

    return net_init, net_apply


def conv_net_unit(in_channels: int,
                  out_channels: int,
                  layers: Tuple[int, ...] or List[int, ...],
                  periodic: bool = False,
                  batch_norm: bool = False,
                  activation: str or Callable = 'ReLU',
                  in_spatial: int or tuple = 2, **kwargs):
    """ Conv-net unit for Invertible Nets"""
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    if isinstance(activation, str):
        activation = ACTIVATIONS[activation]

    init_fn, apply_fn = {}, {}
    if len(layers) < 1:
        layers.append(out_channels)
    init_fn['conv_in'], apply_fn['conv_in'] = stax.serial(
        CONV[d](layers[0], (3,) * d, padding='valid'),
        stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
        activation)
    for i in range(1, len(layers)):
        init_fn[f'conv{i}'], apply_fn[f'conv{i}'] = stax.serial(
            CONV[d](layers[i], (3,) * d, padding='valid'),
            stax.BatchNorm(axis=tuple(range(d + 1))) if batch_norm else stax.Identity,
            activation)

    init_fn['conv_out'], apply_fn['conv_out'] = CONV[d](out_channels, (1,) * d)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        shape, params['conv_in'] = init_fn['conv_in'](rngs[0], input_shape)

        for i in range(1, len(layers)):
            shape, params[f'conv{i + 1}'] = init_fn[f'conv{i + 1}'](rngs[i], shape)

        shape, params['conv_out'] = init_fn['conv_out'](rngs[len(layers)], shape)

        return shape, params

    def net_apply(params, inputs):
        x = inputs

        pad_tuple = [(0, 0)]
        for i in range(d):
            pad_tuple.append((1, 1))
        pad_tuple.append((0, 0))

        out = jnp.pad(x, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')

        out = apply_fn['conv_in'](params['conv_in'], out)

        for i in range(1, len(layers)):
            out = jnp.pad(out, pad_width=pad_tuple, mode='wrap' if periodic else 'constant')
            out = apply_fn[f'conv{i + 1}'](params[f'conv{i + 1}'], out)

        out = apply_fn['conv_out'](params['conv_out'], out)

        return out

    return net_init, net_apply


def u_net_unit(in_channels: int,
               out_channels: int,
               levels: int = 4,
               filters: int or tuple or list = 16,
               batch_norm: bool = True,
               activation='ReLU',
               periodic=False,
               in_spatial: tuple or int = 2,
               use_res_blocks: bool = False, **kwargs):
    """ U-net unit for Invertible Nets"""
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (1,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    # Create layers
    if use_res_blocks:
        inc_init, inc_apply = resnet_block(in_channels, filters[0], periodic, batch_norm, activation, d)
    else:
        inc_init, inc_apply = create_double_conv(d, filters[0], filters[0], batch_norm, activation, periodic)
    init_functions, apply_functions = {}, {}
    for i in range(1, levels):
        if use_res_blocks:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = resnet_block(filters[i - 1], filters[i], periodic, batch_norm, activation, d)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = resnet_block(filters[i] + filters[i - 1], filters[i - 1], periodic, batch_norm, activation, d)
        else:
            init_functions[f'down{i}'], apply_functions[f'down{i}'] = create_double_conv(d, filters[i], filters[i], batch_norm, activation, periodic)
            init_functions[f'up{i}'], apply_functions[f'up{i}'] = create_double_conv(d, filters[i - 1], filters[i - 1], batch_norm, activation, periodic)
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

    return net_init, net_apply


def res_net_unit(in_channels: int,
                 out_channels: int,
                 layers: Tuple[int, ...] or List[int],
                 batch_norm: bool = False,
                 activation: str or Callable = 'ReLU',
                 periodic=False,
                 in_spatial: int or tuple = 2, **kwargs):
    """ Res-net unit for Invertible Nets"""
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    stax_layers = []
    if len(layers) < 1:
        layers.append(out_channels)
    stax_layers.append(resnet_block(in_channels, layers[0], periodic, batch_norm, activation, d))
    for i in range(1, len(layers)):
        stax_layers.append(resnet_block(layers[i - 1], layers[i], periodic, batch_norm, activation, d))
    stax_layers.append(CONV[d](out_channels, (1,) * d))
    return stax.serial(*stax_layers)


NET = {'u_net': u_net_unit, 'res_net': res_net_unit, 'conv_net': conv_net_unit}


def coupling_layer(in_channels: int,
                   activation: str or Callable = 'ReLU',
                   batch_norm: bool = False,
                   in_spatial: int or tuple = 2,
                   net: str = 'u_net',
                   reverse_mask: bool = False,
                   **kwargs):
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    init_fn, apply_fn = {}, {}
    if d == 0:
        init_fn['s1'], apply_fn['s1'] = stax.serial(
            Dense_resnet_block(in_channels, in_channels, batch_norm, activation),
            stax.Tanh)
        init_fn['t1'], apply_fn['t1'] = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)

        init_fn['s2'], apply_fn['s2'] = stax.serial(
            Dense_resnet_block(in_channels, in_channels, batch_norm, activation),
            stax.Tanh)
        init_fn['t2'], apply_fn['t2'] = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)
    else:
        init_fn['s1'], apply_fn['s1'] = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[], batch_norm=batch_norm, activation=activation, in_spatial=in_spatial, **kwargs)
        init_fn['t1'], apply_fn['t1'] = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[], batch_norm=batch_norm, activation=activation, in_spatial=in_spatial, **kwargs)
        init_fn['s2'], apply_fn['s2'] = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[], batch_norm=batch_norm, activation=activation, in_spatial=in_spatial, **kwargs)
        init_fn['t2'], apply_fn['t2'] = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[], batch_norm=batch_norm, activation=activation, in_spatial=in_spatial, **kwargs)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)

        shape, params['s1'] = init_fn['s1'](rngs[0], input_shape)
        shape, params['t1'] = init_fn['t1'](rngs[1], input_shape)
        shape, params['s2'] = init_fn['s2'](rngs[2], input_shape)
        shape, params['t2'] = init_fn['t2'](rngs[3], input_shape)

        return shape, params

    def net_apply(params, inputs, invert=False):
        x = inputs

        mask = get_mask(x, reverse_mask, 'NCHW')

        if invert:
            v1 = x * mask
            v2 = x * (1 - mask)

            s1 = apply_fn['s1'](params['s1'], v1)
            t1 = apply_fn['t1'](params['t1'], v1)

            u2 = (1 - mask) * (v2 - t1) * jnp.exp(-jnp.tanh(s1))

            s2 = apply_fn['s2'](params['s2'], u2)
            t2 = apply_fn['t2'](params['t2'], u2)

            u1 = mask * (v1 - t2) * jnp.exp(-jnp.tanh(s2))

            return u1 + u2
        else:
            u1 = x * mask
            u2 = x * (1 - mask)

            s2 = apply_fn['s2'](params['s2'], u2)
            t2 = apply_fn['t2'](params['t2'], u2)

            v1 = mask * (u1 * jnp.exp(jnp.tanh(s2)) + t2)

            s1 = apply_fn['s1'](params['s1'], v1)
            t1 = apply_fn['t1'](params['t1'], v1)

            v2 = (1 - mask) * (u2 * jnp.exp(jnp.tanh(s1)) + t1)

            return v1 + v2

    return net_init, net_apply


def invertible_net(in_channels: int,
                   num_blocks: int,
                   batch_norm: bool = False,
                   net: str = 'u_net',
                   activation: str or type = 'ReLU',
                   in_spatial: tuple or int = 2, **kwargs):
    """
    ΦFlow also provides invertible neural networks that are capable of inverting the output tensor back to the input tensor initially passed.\ These networks have far reaching applications in predicting input parameters of a problem given its observations.\ Invertible nets are composed of multiple concatenated coupling blocks wherein each such block consists of arbitrary neural networks.

    Currently, these arbitrary neural networks could be set to u_net(default), conv_net, res_net or dense_net blocks with in_channels = out_channels.
    The architecture used is popularized by ["Real NVP"](https://arxiv.org/abs/1605.08803).

    Arguments:

        in_channels : input channels of the feature map, dtype : int
        num_blocks : number of coupling blocks inside the invertible net, dtype : int
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int
        net : type of neural network blocks used in coupling layers, dtype : str
        **kwargs : placeholder for arguments not supported by the function

    Returns:

        Invertible Net model as specified by input arguments

    Note: Currently supported values for net are 'u_net'(default), 'conv_net' and 'res_net'.
    For choosing 'dense_net' as the network block in coupling layers in_spatial must be set to zero.
    """
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    init_fn, apply_fn = {}, {}
    for i in range(num_blocks):
        init_fn[f'CouplingLayer{i + 1}'], apply_fn[f'CouplingLayer{i + 1}'] = coupling_layer(in_channels, activation, batch_norm, d, net, (i % 2 == 0), **kwargs)

    def net_init(rng, input_shape):
        params = {}
        rngs = random.split(rng, 2)
        for i in range(num_blocks):
            shape, params[f'CouplingLayer{i + 1}'] = init_fn[f'CouplingLayer{i + 1}'](rngs[i], input_shape)
        return shape, params

    def net_apply(params, inputs, invert=False):
        out = inputs
        if invert:
            for i in range(num_blocks, 0, -1):
                out = apply_fn[f'CouplingLayer{i}'](params[f'CouplingLayer{i}'], out, invert)
        else:
            for i in range(1, num_blocks + 1):
                out = apply_fn[f'CouplingLayer{i}'](params[f'CouplingLayer{i}'], out)
        return out

    if d == 0:
        net = StaxNet(net_init, net_apply, (1,) + (in_channels,))
    else:
        net = StaxNet(net_init, net_apply, (1,) + (1,) * d + (in_channels,))
    net.initialize()
    return net
