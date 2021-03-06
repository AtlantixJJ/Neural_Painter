"""
Complex tensorflow operations, as a layer.
"""
import tensorflow as tf
from tensorflow.contrib import layers as L
from lib import ops, utils

def default_batch_norm(name, inputs, training=True, phase='default', reuse=False, epsilon=0.001, momentum=0.99):
    with tf.variable_scope(name, reuse=reuse):
        size = inputs.get_shape().as_list()[-1]

        gamma = tf.get_variable('gamma', [size], initializer=tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', [size], initializer=tf.zeros_initializer(), trainable=True)

        #with tf.variable_scope(phase, reuse=tf.AUTO_REUSE):
        population_mean = tf.get_variable(
            'moving_mean', [size],
            initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(
            'moving_var', [size],
            initializer=tf.ones_initializer(), trainable=False)

        if len(inputs.get_shape()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            gamma = tf.reshape(gamma, [1, 1, 1, size])
            beta = tf.reshape(gamma, [1, 1, 1, size])
        elif len(inputs.get_shape()) == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            gamma = tf.reshape(gamma, [1, size])
            beta = tf.reshape(gamma, [1, size])

        train_mean_op = tf.assign(
            population_mean,
            population_mean * momentum + batch_mean * (1 - momentum),
            name="assign_mean_" + phase)
        train_var_op = tf.assign(
            population_var,
            population_var * momentum + batch_var * (1 - momentum),
            name="assign_var_" + phase)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var_op)

    return tf.cond(training,
        lambda: tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon),
        lambda: tf.nn.batch_normalization(inputs, population_mean, population_var, beta, gamma, epsilon)
        )

    #return tf.nn.batch_normalization(inputs, population_mean, population_var, beta, gamma, epsilon)

def conditional_batch_normalization(name, inputs, conditions, training=True, phase='default', update_collection=None, is_project=True, reuse=False, epsilon=0.001, momentum=0.99):
    """
    conditions: [gamma, beta].
    spectral_norm & update_collection: for spectral weight normalization
    """

    with tf.variable_scope(name, reuse=reuse):
        size = inputs.get_shape().as_list()[-1]
        p_dim = size if is_project else 1
        """
        w_gamma = tf.get_variable("weight_gamma", shape=[conditions.get_shape()[-1], p_dim],
            initializer=tf.initializers.random_normal(stddev=2), trainable=True)
        w_beta = tf.get_variable("weight_beta", shape=[conditions.get_shape()[-1], p_dim],
            initializer=tf.initializers.random_normal(stddev=2), trainable=True)
        gamma = 1 + tf.matmul(conditions, w_gamma)
        beta = tf.matmul(conditions, w_beta)
        """

        gamma = tf.get_variable('gamma', [size], initializer=tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', [size], initializer=tf.zeros_initializer(), trainable=True)

        #with tf.variable_scope(phase, reuse=tf.AUTO_REUSE):
        population_mean = tf.get_variable(
            'moving_mean', [size],
            initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(
            'moving_var', [size],
            initializer=tf.ones_initializer(), trainable=False)

        if len(inputs.get_shape()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            gamma = tf.reshape(gamma, [-1, 1, 1, p_dim])
            beta = tf.reshape(beta, [-1, 1, 1, p_dim])
        elif len(inputs.get_shape()) == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            gamma = tf.reshape(gamma, [-1, p_dim])
            beta = tf.reshape(beta, [-1, p_dim])

        train_mean_op = tf.assign(
            population_mean,
            population_mean * momentum + batch_mean * (1 - momentum),
            name="assign_mean_" + phase)
        train_var_op = tf.assign(
            population_var,
            population_var * momentum + batch_var * (1 - momentum),
            name="assign_var_" + phase)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var_op)

    return tf.cond(training,
        lambda: tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon),
        lambda: tf.nn.batch_normalization(inputs, population_mean, population_var, beta, gamma, epsilon)
        )

    #return tf.nn.batch_normalization(inputs, population_mean, population_var, beta, gamma, epsilon)

def simple_batch_norm(name, inputs, phase='default', training=True, reuse=False, epsilon=1e-6, decay=0.9):
    """
    moving average, without any trainable variables
    """

    with tf.variable_scope(name, reuse=reuse):
        size = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(phase, reuse=tf.AUTO_REUSE):
            population_mean = tf.get_variable(
                'moving_mean', [size],
                initializer=tf.zeros_initializer(), trainable=False)
            population_var = tf.get_variable(
                'moving_var', [size],
                initializer=tf.ones_initializer(), trainable=False)

        if len(inputs.get_shape()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        elif len(inputs.get_shape()) == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        
        cond_new_pm = tf.cond(training,
            true_fn=lambda: population_mean * decay + batch_mean * (1 - decay),
            false_fn=lambda: population_mean)
        cond_new_pv = tf.cond(training,
            true_fn=lambda: population_var * decay + batch_var * (1 - decay),
            false_fn=lambda: population_var)

        train_mean_op = tf.assign(population_mean, cond_new_pm)
        train_var_op = tf.assign(population_var, cond_new_pv)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var_op)

        #with tf.control_dependencies([train_mean_op, train_var_op]):
        x = tf.nn.batch_normalization(inputs, population_mean, population_var, None, None, epsilon)
        return x

def caffe_batch_norm(name, inputs, phase='default', training=True, reuse=False, epsilon=1e-6, decay=0.9):
    """
    moving average, without any trainable variables.
    And do not calculate gradient w.r.t mu and sigma.
    """

    with tf.variable_scope(name, reuse=reuse):
        size = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(phase, reuse=tf.AUTO_REUSE):
            population_mean = tf.get_variable(
                'moving_mean', [size],
                initializer=tf.zeros_initializer(), trainable=False)
            population_var = tf.get_variable(
                'moving_var', [size],
                initializer=tf.ones_initializer(), trainable=False)

        if len(inputs.get_shape()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        elif len(inputs.get_shape()) == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        
        # do not calculate gradient on mean and var
        batch_mean = tf.stop_gradient(batch_mean)
        batch_var = tf.stop_gradient(batch_var)
        
        cond_new_pm = tf.cond(training,
            true_fn=lambda: population_mean * decay + batch_mean * (1 - decay),
            false_fn=lambda: population_mean)
        cond_new_pv = tf.cond(training,
            true_fn=lambda: population_var * decay + batch_var * (1 - decay),
            false_fn=lambda: population_var)

        train_mean_op = tf.assign(population_mean, cond_new_pm)
        train_var_op = tf.assign(population_var, cond_new_pv)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var_op)

        #with tf.control_dependencies([train_mean_op, train_var_op]):
        x = tf.nn.batch_normalization(inputs, population_mean, population_var, None, None, epsilon)
        return x

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

def conv2d(name, input, output_dim, filter_size=3, stride=1,
    spectral=False, reuse=False):
    """
    Args:
        spectral: If to use spectral normalization
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("kernel", shape=[filter_size, filter_size, input.get_shape()[-1], output_dim], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(.0))

        input_dim = input.get_shape()[-1]

        if spectral > 0:
            if spectral == 1: # spectral norm weight from noise vector
                u = None
            elif spectral == 2: #spectral norm weight from data vector
                u = tf.image.random_crop(input, [filter_size, filter_size, input_dim])
                u = tf.reshape(u, [1, -1])
            w = ops.spectral_normed_weight(w, u=u)

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
        
        H, W, C = input.get_shape().as_list()[1:]

        if spectral > 0:
            if spectral == 1: # spectral norm weight from noise vector
                u = None
            elif spectral == 2: #spectral norm weight from data vector
                u = tf.image.random_crop(input, [1, 1, C])
                u = tf.reshape(u, [1, -1])
            w_ = tf.transpose(w, [0, 1, 3, 2])
            w = ops.spectral_normed_weight(w_, u=u, update_collection=update_collection)
            w = tf.transpose(w_, [0, 1, 3, 2])

        x = tf.nn.conv2d_transpose(input,
            filter=w,
            output_shape=[tf.shape(input)[0], H*stride, W*stride, output_dim],
            strides=[1, stride, stride, 1],
            padding="SAME") + b

    return x

def linear(name, input, output_dim, spectral=0, reuse=False):
    """
    Get a linear layer,while the initializer is a random_uniform_initializer.
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("weight", shape=[input.get_shape()[-1], output_dim], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(.0))

        if spectral > 0:
            if spectral == 1: # spectral norm weight from noise vector
                u = None
            elif spectral == 2: #spectral norm weight from data vector
                u = input[0:1]
            w = ops.spectral_normed_weight(w, u=u)

        x = tf.matmul(input, w) + b
        #with tf.device("/device:CPU:0"):
        #    x = tf.Print(x, [tf.reduce_sum(tf.abs(w))])
    
    return x

def subpixel_conv(name, input, output_dim, filter_size=3, spectral=False, reuse=False):
    output = conv2d(name, input, 4*output_dim, filter_size, 1, 
        spectral=spectral, reuse=reuse)
    return tf.depth_to_space(output, 2)

def simple_residual_block(name, x, filter_size=3,
    activation_fn=tf.nn.relu, norm_fn=None,
    spectral=False, reuse=False):
    """
    identity + 2 conv with same depth, bottleneck
    """

    input_dim = x.get_shape().as_list()[-1]
    x_skip = tf.identity(x)

    x = norm_fn(name + "/bn1", x)
    x = activation_fn(x)
    x = conv2d(name + "/conv1", x, input_dim // 2, filter_size, 1,
        spectral=spectral, reuse=reuse)

    x = norm_fn(name + "/bn2", x)
    x = activation_fn(x)
    x = conv2d(name + "/conv2", x, input_dim, filter_size, 1,
        spectral=spectral, reuse=reuse)

    return x_skip + x

def upsample_residual_block(name, x, dim, activation_fn=tf.nn.relu, norm_fn=None,
    spectral=1, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        h, w, c = x.get_shape()[1:]

        #enhance identity
        x_skip = tf.image.resize_nearest_neighbor(x, (h * 2, w * 2))
        if c != dim:
            x_skip = conv2d("skip/conv", x_skip, dim, 1, 1, spectral, reuse)
        
        #x = tf.image.resize_bilinear(x, (h * 2, w * 2))
        x = norm_fn("bn1", x)
        x = activation_fn(x)
        x = tf.image.resize_nearest_neighbor(x, (h * 2, w * 2))
        x = conv2d(name + "/conv1", x, c, 3, 1, spectral, reuse)

        x = norm_fn("bn2", x)
        x = activation_fn(x)
        x = conv2d("conv2", x, dim, 3, 1, spectral, reuse)

    return x + x_skip

def downsample_residual_block(name, x, dim, activation_fn=tf.nn.relu, norm_fn=None,
    spectral=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        c = x.get_shape()[-1]

        # enhance identity
        if c != dim:
            x_skip = conv2d("skip/conv", x, dim, 1, 1, spectral, reuse)
            x_skip = tf.nn.avg_pool(x_skip, 2, 2, "VALID")
        else:
            x_skip = tf.nn.avg_pool(x, 2, 2, "VALID")

        x = norm_fn("bn1", x)
        x = activation_fn(x)
        x = conv2d("conv1", x, dim // 2, 3, 1, spectral, reuse)

        x = norm_fn("bn2", x)
        x = activation_fn(x)
        x = conv2d("conv2", x, dim, 3, 1, spectral, reuse)
        x = tf.nn.avg_pool(x, 2, 2, "VALID")

    return x + x_skip

def attention(name, x, ch, spectral_norm=True, reuse=False):
    f = conv2d(name + "f_conv", x, ch // 8, 1, 1, spectral_norm) # [bs, h, w, c']
    g = conv2d(name + "g_conv", x, ch // 8, 1, 1, spectral_norm) # [bs, h, w, c']
    h =  conv2d(name + "h_conv", x, ch, 1, 1, spectral_norm) # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(ops.hw_flatten(g), ops.hw_flatten(f), transpose_b=True) # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, ops.hw_flatten(h)) # [bs, N, C]
    with tf.variable_scope(name, reuse=reuse):
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    x = gamma * o + x

    return x

def get_norm(name, x, method, training=None, reuse=False):
    """
    [Deprecated]
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