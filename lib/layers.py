import tensorflow as tf

from tensorflow.contrib import layers as L
from lib import ops

def default_batch_norm(inputs, training, data_format=""):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.compat.v1.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def instance_normalization(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def group_normalization(x, G=32, name='group', reuse=False):
    esp = 1e-5
    with tf.variable_scope(name, reuse=reuse):
        # normalize
        # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        
        x = tf.reshape(x, shape=[-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                                initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

def get_norm(name, x, method, training=None, reuse=False):
    """
    Get batch normalization of different kind.
    """
    if method is None:
        return x
    
    if method.find("inst") > -1:
        # usually fail to produce any realistic result in GAN
        return instance_normalization(x)
    elif method.find("default") > -1:
        with tf.variable_scope(name, reuse=reuse):
            return default_batch_norm(x, training)
        #return tf.layers.batch_normalization(inputs=x, name=name + "_" + method, reuse=reuse, training=training, epsilon=1e-6)
    elif method.find("contrib") > -1:
        # Common choice
        return tf.contrib.layers.batch_norm(inputs=x, scope=name + "_" + method, reuse=reuse, is_training=training, epsilon=1e-6)
    elif method.find("group") > -1:
        return group_normalization(x, name=name + "_" + method, reuse=reuse)
    else:
        return x

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def learned_sum(name, input, reuse=False):
    """
    Improvement of Global Average Pooling.
    """
    with tf.variable_scope(name, reuse=reuse):
        l = input.get_shape().as_list()[1:]
        avg = 1. / float(l[0] * l[1])
        mulmap = tf.get_variable("coef", l[:-1]+[1,], initializer=tf.constant_initializer(avg))
        #addbias = tf.get_variable("addbias", [], initializer=tf.constant_initializer(0.0))
        output = tf.reduce_sum(input * mulmap, axis=[1, 2])
    return output

def conditional_batch_normalization(name, inputs, conditions, training=True, is_project=True, reuse=False, epsilon=1e-6, decay=0.99):
    """
    conditions: [gamma, beta]
    """

    with tf.variable_scope(name, reuse=reuse):
        size = inputs.get_shape().as_list()[-1]
        p_dim = size if is_project else 1

        w = tf.get_variable("weight", shape=[conditions.get_shape()[-1], p_dim * 2], initializer=tf.orthogonal_initializer)

        projections = tf.reshape(tf.matmul(conditions, w), [-1, 2, p_dim])

        population_mean = tf.get_variable(
            'moving_mean', [size],
            initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(
            'moving_var', [size],
            initializer=tf.ones_initializer(), trainable=False)

        if len(inputs.get_shape()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            gamma = tf.reshape(projections[:, 0], [-1, 1, 1, p_dim])
            beta = tf.reshape(projections[:, 1], [-1, 1, 1, p_dim])
        elif len(inputs.get_shape()) == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            gamma = tf.reshape(conditions[:, 0], [-1, p_dim])
            beta = tf.reshape(conditions[:, 1], [-1, p_dim])

        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var_op)

        x = tf.nn.batch_normalization(inputs, population_mean, population_var, None, None, epsilon)
        return gamma * x + beta

def conv2d(name, input, output_dim, filter_size=3, stride=1,
    spectral=False, update_collection=None, reuse=False):
    """
    Args:
        spectral: If to use spectral normalization
        update_collection: only used in spectral normalization
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("kernel", shape=[filter_size, filter_size, input.get_shape()[-1], output_dim], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(.0))

        if spectral: w = ops.spectral_normed_weight(w, update_collection=update_collection)

        x = tf.nn.conv2d(input, filter=w, strides=[1, stride, stride, 1], padding="SAME") + b

    return x

def deconv2d(name, input, output_dim, filter_size=3, stride=1,
    spectral=False, update_collection=None, reuse=False):
    """
    Args:
        spectral: If to use spectral normalization
        update_collection: only used in spectral normalization
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("kernel", shape=[filter_size, filter_size, output_dim,  input.get_shape()[-1]], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0))

        if spectral: w = ops.spectral_normed_weight(w, update_collection=update_collection)
        
        H, W = input.get_shape().as_list()[1:3]

        x = tf.nn.conv2d_transpose(input,
            filter=w,
            output_shape=[tf.shape(input)[0], H*stride, W*stride, output_dim],
            strides=[1, stride, stride, 1],
            padding="SAME") + b

    return x

def linear(name, input, output_dim, spectral=False, update_collection=None, reuse=False):
    """
    Get a linear layer,while the initializer is a random_uniform_initializer.
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("weight", shape=[input.get_shape()[-1], output_dim], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(.0))

        if spectral: w = ops.spectral_normed_weight(w, update_collection=update_collection)

        x = tf.matmul(input, w) + b
    
    return x

def subpixel_conv(name, input, output_dim, filter_size=3, spectral=False, update_collection=None, reuse=False):
    output = conv2d(name, input, 4*output_dim, filter_size, 1, 
        spectral=spectral, update_collection=update_collection, reuse=reuse)
    return tf.depth_to_space(output, 2)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def simple_residual_block(name, input, filter_size=3,
    activation_fn=tf.nn.relu, norm_fn=None,
    spectral=False, update_collection=None, reuse=False):
    """
    identity + 2 conv with same depth
    """

    input_dim = input.get_shape().as_list()[-1]

    x = norm_fn(name + "/bn1", x)
    x = activation_fn(x)
    x = conv2d(name + "/conv1", input, input_dim, filter_size, 1,
        spectral=spectral, update_collection=update_collection, reuse=reuse)

    x = norm_fn(name + "/bn2", x)
    x = activation_fn(x)
    x = conv2d(name + "/conv2", x, input_dim, filter_size, 1,
        spectral=spectral, update_collection=update_collection, reuse=reuse)

    return input + x

def upsample_residual_block(name, x, dim, activation_fn=tf.nn.relu, norm_fn=None,
    spectral=False, update_collection=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        base = tf.identity(x)
        h, w, c = base.get_shape()[1:]

        #x_skip = deconv2d(name + "/skip", base, dim, 4, 2, spectral, update_collection, reuse)
        x_skip = conv2d(name + "/skip", base, dim, 1, 1, spectral, update_collection, reuse)
        x_skip = tf.image.resize_bilinear(x_skip, (h * 2, w * 2))

        x = norm_fn(name + "/bn1", x)
        x = activation_fn(x)
        x = tf.image.resize_bilinear(x, (h * 2, w * 2))
        x = conv2d(name + "/conv1", x, dim, 3, 1, spectral, update_collection, reuse)
        #x = deconv2d(name + "/conv1", x, dim, 4, 2, spectral, update_collection, reuse)

        x = norm_fn(name + "/bn2", x)
        x = activation_fn(x)
        x = conv2d(name + "/conv2", x, dim, 3, 1, spectral, update_collection)

    return x + x_skip

def downsample_residual_block(name, x, dim, activation_fn=tf.nn.relu, norm_fn=None,
    spectral=False, update_collection=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        base = tf.identity(x)

        x_skip = conv2d(name + "/skip", base, dim, 1, 1, spectral, update_collection, reuse)
        x_skip = tf.nn.avg_pool(x_skip, 2, 2, "VALID")

        x = norm_fn(name + "/bn1", x)
        x = activation_fn(x)
        x = conv2d(name + "/conv1", x, dim, 3, 1, spectral, update_collection, reuse)

        x = norm_fn(name + "/bn2", x)
        x = activation_fn(x)
        x = conv2d(name + "/conv2", x, dim, 3, 1, spectral, update_collection)
        x = tf.nn.avg_pool(x, 2, 2, "VALID")

    return x + x_skip

def attention(name, x, ch, spectral_norm=True, update_collection=None, reuse=False):
    f = conv2d(name + "f_conv", x, ch // 8, 1, 1, spectral_norm, update_collection) # [bs, h, w, c']
    g = conv2d(name + "g_conv", x, ch // 8, 1, 1, spectral_norm, update_collection) # [bs, h, w, c']
    h =  conv2d(name + "h_conv", x, ch, 1, 1, spectral_norm, update_collection) # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    with tf.variable_scope(name, reuse=reuse):
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    x = gamma * o + x

    return x

"""
def attention_2(name, x, ch, spectral_norm=True, update_collection=None, reuse=False):
    batch_size, height, width, num_channels = x.get_shape().as_list()
    f = conv2d(name + "f_conv", x, ch // 8, 1, 1, spectral_norm, update_collection)  # [bs, h, w, c']
    f = max_pooling(f)

    g = conv(x, ch // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']

    h = conv(x, ch // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
    h = max_pooling(h)

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
    o = conv(o, ch, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
    x = gamma * o + x

    return x
"""