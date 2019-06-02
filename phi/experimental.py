import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from phi.flow import *


def conv_function(scope, constants_file=None):
    if constants_file is not None:
        reader = pywrap_tensorflow.NewCheckpointReader(constants_file)
        def conv(n, filters, kernel_size, strides=[1,1,1,1], padding="VALID", activation=None, name=None, kernel_initializer=None):
            assert name != None
            kernel = reader.get_tensor("%s/%s/kernel"%(scope,name))
            assert kernel.shape[-1] == filters, "Expected %d filters but loaded kernel has shape %s for conv %s" % (kernel_size, kernel.shape, name)
            if isinstance(kernel_size, int):
                assert kernel.shape[0] == kernel.shape[1] == kernel_size
            else:
                assert kernel.shape[0:2] == kernel_size
            if isinstance(strides, int):
                strides = [1, strides, strides, 1]
            elif len(strides) == 2:
                strides = [1, strides[0], strides[1], 1]
            n = tf.nn.conv2d(n, kernel, strides=strides, padding=padding.upper(), name=name)
            if activation is not None:
                n = activation(n)
            n = tf.nn.bias_add(n, reader.get_tensor("%s/%s/bias"%(scope,name)))
            return n
    else:
        def conv(n, filters, kernel_size, strides=(1,1), padding="valid", activation=None, name=None, kernel_initializer=None):
            with tf.variable_scope(scope):
                return tf.layers.conv2d(n, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        activation=activation, name=name, reuse=tf.AUTO_REUSE, kernel_initializer=kernel_initializer)
    return conv


def restore_net(scope, sess, graph, path):
    vars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, path)



def normalize_dipole(div_dipole):
    div = div_dipole[...,0]
    dipoles = div_dipole[...,1:]

    mean, var = tf.nn.moments(div, axes=[1, 2])  # [batch_size, 1]
    mean = math.reshape(mean, [-1, 1, 1, 1])
    std = math.reshape(tf.sqrt(var), [-1, 1, 1, 1])
    std = math.maximum(std, 1e-5)
    return (div_dipole - mean) / std, std


def normalize(tensor):
    mean, var = tf.nn.moments(tensor, axes=[1, 2])  # [batch_size, 1]
    mean = math.reshape(mean, [-1, 1, 1, 1])
    std = math.reshape(tf.sqrt(var), [-1, 1, 1, 1])
    std = math.maximum(std, 1e-5)
    return (tensor - mean) / std


def std(tensor, min=1e-20):
    mean, var = tf.nn.moments(tensor, axes=[1, 2, 3])  # [batch_size, 1]
    std = math.reshape(tf.sqrt(var), [-1])
    std = math.maximum(std, min)
    return std


def to_dipole_format(tensor):
    # vectors should be q, px, py
    rank = spatial_rank(tensor)
    if tensor.shape[-1] == 1:
        return math.pad(tensor, [[0,0]]*(rank+1)+[[0,rank]])
    if tensor.shape[-1] == spatial_rank(tensor)+1:
        return tensor
    raise ValueError("Cannot convert tensor with shape {} to dipole format.".format(tensor.shape))


def com_downsample2x(tensor, sum=True):
    tensor = to_dipole_format(tensor)
    rank = spatial_rank(tensor)
    f = 1 if sum else 0.5
    dims = range(spatial_rank(tensor))
    for dim in dims:
        upper_slices = [(slice(1, None, 2) if i == dim else slice(None)) for i in dims]
        lower_slices = [(slice(0, None, 2) if i == dim else slice(None)) for i in dims]
        s = (tensor[[slice(None)] + upper_slices + [slice(0, 1)]] + tensor[[slice(None)] + lower_slices + [slice(0, 1)]]) * f
        components = [s]  # ordered like q, px, py, pz
        for cdim in dims:
            p_component = [slice(rank - cdim, rank - cdim + 1)]
            s_left =  tensor[[slice(None)] + lower_slices + [slice(0, 1)]]
            s_right = tensor[[slice(None)] + upper_slices + [slice(0, 1)]]
            p_left =  tensor[[slice(None)] + lower_slices + p_component]
            p_right = tensor[[slice(None)] + upper_slices + p_component]
            if cdim == dim: # Contract dimension, calculate p from previous level
                p = ( s_left * (p_left - 1) + s_right * (p_right + 1) ) / (2 * math.maximum(s, 1e-20))
            else: # Take the weighted average in other other components
                p = (s_left * p_left + s_right * p_right) / math.maximum(s, 1e-20)
            components.insert(1, p)
        tensor = math.concat(components, axis=-1)
    return tensor


def downsample_dipole_2d_2x(tensor, scaling="average"):
    tensor = to_dipole_format(tensor)
    # [filter_height, filter_width, in_channels, out_channels]
    filter = np.zeros([2, 2, 3, 3], np.float32)
    s = 1 if scaling == "sum" else 0.5

    # q
    filter[:, : , 0, 0] = 1 * s**2
    # px
    filter[:, 0 , 0, 1] = -1 * s
    filter[:, 1 , 0, 1] = 1 * s
    filter[:, : , 1, 1] = 0.5 * s
    # py
    filter[0, : , 0, 2] = -1 * s
    filter[1, : , 0, 2] = 1 * s
    filter[:, : , 2, 2] = 0.5 * s
    return tf.nn.conv2d(tensor, filter, strides=[1, 2, 2, 1], padding="SAME")


def moment_downsample2x(tensor, sum=False):
    tensor = to_dipole_format(tensor)
    rank = spatial_rank(tensor)
    s = 1 if sum else 0.5
    dims = range(spatial_rank(tensor))
    for dim in dims: # z,y,x
        entry = rank - dim
        upper_slices = [(slice(1, None, 2) if i == dim else slice(None)) for i in dims]
        lower_slices = [(slice(0, None, 2) if i == dim else slice(None)) for i in dims]
        q = (tensor[[slice(None)] + upper_slices + [slice(0,1)]] + tensor[[slice(None)] + lower_slices + [slice(0,1)]]) * s
        p = (tensor[[slice(None)] + upper_slices + [slice(0,1)]] - tensor[[slice(None)] + lower_slices + [slice(0,1)]]) * s
        p_dim_slice = [slice(entry, entry+1)]
        p += (tensor[[slice(None)] + upper_slices + p_dim_slice] + tensor[[slice(None)] + lower_slices + p_dim_slice]) * s * 0.5
        components = [ q ] # ordered like q, px, py, pz
        for odim in dims:
            if odim != dim:
                oentry = rank - odim
                o_dim_slice = [slice(oentry, oentry+1)]
                # take the average
                o = (tensor[[slice(None)] + upper_slices + o_dim_slice] + tensor[[slice(None)] + lower_slices + o_dim_slice]) * s
                components.insert(1,o)
            else:
                components.insert(1,p)
        tensor = math.concat(components, axis=-1)
    return tensor



def upsample_flatten_dipole_2d_2x(tensor, scaling="average"): # for non-scaling fields
    w, h = int(tensor.shape[2]), int(tensor.shape[1])
    s = 0.5 if scaling == "sum" else 1
    # [filter_height, filter_width, in_channels, out_channels]
    filter = np.zeros([1, 1, 3, 4], np.float32)
    # q
    filter[:, :, 0, :] = s**2
    # px
    filter[0, 0, 1, (0,2)] = -0.25
    filter[0, 0, 1, (1,3)] = 0.25
    # py
    filter[0, 0, 2, (0,1)] = -0.25
    filter[0, 0, 2, (2,3)] = 0.25
    tensor = tf.nn.conv2d(tensor, filter, strides=[1, 1, 1, 1], padding="SAME")
    upper = math.reshape(tensor[:, :, :, 0:2], [-1, h, w  * 2, 1])
    lower = math.reshape(tensor[:, :, :, 2:4], [-1, h, w * 2, 1])
    tensor = math.stack([upper, lower], axis=-1)
    tensor = tf.transpose(tensor, [0, 1, 4, 3, 2])
    tensor = math.reshape(tensor, [-1, h * 2, w * 2, 1])
    return tensor


def upsample_dipole_2d_2x(tensor, scaling="average"): # for non-scaling fields
    w, h = int(tensor.shape[2]), int(tensor.shape[1])
    # [filter_height, filter_width, in_channels, out_channels]
    filter = np.zeros([1, 1, 3, 4, 3], np.float32)
    # q from q
    if scaling == "average":
        filter[:, :, 0, :, 0] = 1
    elif scaling == "sum":
        filter[:, :, 0, :, 0] = 0.25
    # q from px
    filter[0, 0, 1, (0,2), 0] = -0.2
    filter[0, 0, 1, (1,3), 0] = 0.2
    # q from py
    filter[0, 0, 2, (0,1), 0] = -0.2
    filter[0, 0, 2, (2,3), 0] = 0.2
    # px
    filter[:, :, 1, :, 1] = 0.5
    # py
    filter[:, :, 2, :, 2] = 0.5
    filter = filter.reshape([1, 1, 3, 12])

    tensor = tf.nn.conv2d(tensor, filter, strides=[1, 1, 1, 1], padding="SAME")
    tensor = tf.reshape(tensor, [-1, h, w, 4, 3])
    upper = math.reshape(tensor[:, :, :, 0:2, :], [-1, h, w  * 2, 1, 3])
    lower = math.reshape(tensor[:, :, :, 2:4, :], [-1, h, w * 2, 1, 3])
    tensor = math.stack([upper, lower], axis=-2)
    tensor = tf.transpose(tensor, [0, 1, 4, 3, 2, 5])
    tensor = math.reshape(tensor, [-1, h * 2, w * 2, 3])
    return tensor


def to_gauss_format(tensor):
    # vectors should be q, px, py
    rank = spatial_rank(tensor)
    if tensor.shape[-1] == 1:
        return math.pad(tensor, [[0,0]]*(rank+1)+[[0,rank*2]])
    if tensor.shape[-1] == spatial_rank(tensor)*2+1:
        return tensor
    raise ValueError("Cannot convert tensor with shape {} to dipole format.".format(tensor.shape))


def gauss_downsample2x(tensor, sum=True):
    tensor = to_gauss_format(tensor)
    rank = spatial_rank(tensor)
    f = 1 if sum else 0.5
    dims = range(spatial_rank(tensor))
    for dim in dims:
        upper_slices = [(slice(1, None, 2) if i == dim else slice(None)) for i in dims]
        lower_slices = [(slice(0, None, 2) if i == dim else slice(None)) for i in dims]
        s = (tensor[[slice(None)] + upper_slices + [slice(0, 1)]] + tensor[[slice(None)] + lower_slices + [slice(0, 1)]]) * f
        components = [s]  # ordered like q, com_x, sigma_x, com_y, sigma_y, com_z, sigma_z
        for cdim in dims:
            com_component = [slice(rank*2 - cdim*2 - 1, rank*2 - cdim*2)]
            sigma_component = [slice(rank*2 - cdim*2, rank*2 - cdim*2 + 1)]
            sum_left =   tensor[[slice(None)] + lower_slices + [slice(0, 1)]]
            sum_right =  tensor[[slice(None)] + upper_slices + [slice(0, 1)]]
            com_left =   tensor[[slice(None)] + lower_slices + com_component]
            com_right =  tensor[[slice(None)] + upper_slices + com_component]
            sigma_left = tensor[[slice(None)] + lower_slices + sigma_component]
            sigma_right =tensor[[slice(None)] + upper_slices + sigma_component]
            if cdim == dim: # Contract dimension, calculate com, sigma from previous level
                com = ( sum_left * (com_left - 1) + sum_right * (com_right + 1) ) / (2 * math.maximum(s, 1e-20))
                m2 = ( sum_left * (com_left - 1)**2 + sum_right * (com_right + 1)**2 ) / (4 * math.maximum(s, 1e-20)) # TODO plus sigmas
                sigma = m2 # backend.sqrt(m2 - com**2)
            else: # Take the weighted average in other other components
                com = (sum_left * com_left + sum_right * com_right) / math.maximum(s, 1e-20)
                sigma = (sum_left * sigma_left + sum_right * sigma_right) / math.maximum(s, 1e-20)
            components.insert(1, com)
            components.insert(2, sigma)
        tensor = math.concat(components, axis=-1)

    components = [tensor[...,0]]
    for dim in dims:
        com = tensor[...,dim*2+1]
        m2 = tensor[...,dim*2+2]
        components.append(com)
        components.append(math.sqrt(math.maximum(m2 - com**2, 1e-20)))
    tensor = math.stack(components, axis=-1)
    return tensor