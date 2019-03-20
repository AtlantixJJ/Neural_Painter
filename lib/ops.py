import tensorflow as tf


import numpy as np
import skimage.transform as T
from tensorlayer.prepro import flip_axis, illumination

###### Numpy Operations ######
##############################

def arbitrary_rows_cols(arr, num_rows, num_cols, gray=False):
    """
    Align a np.array to any rows and cols matrix
    """
    def reshape_row(arr):
        return reduce(lambda x, y: np.concatenate((x,y), axis=1), arr)

    def reshape_col(arr):
        return reduce(lambda x, y: np.concatenate((x,y), axis=0), arr)

    num_images, height, width, depth, = arr.shape
    rows = []
    for i in range(num_rows):
        row_image = arr[i*num_cols:i*num_cols+num_cols]
        r_n, r_h, r_w, r_d = row_image.shape
        if row_image.shape[0] != num_cols:
            for _ in range(num_cols - row_image.shape[0]):
                row_image = np.concatenate((row_image, np.expand_dims(np.zeros((height, width, depth)), axis=0)), axis=0)
        row_image = reshape_row(row_image)
        rows.append(row_image)
    mosaic = reshape_col(rows)
    return mosaic
    
def get_preprocess_fn(kind='none'):
    """
    Assume input to be 0~255, get preprocessing function
    """
    def func(x):
        origin_scale = 1
        # detect scale
        max_, min_ = x.max(), x.min()
        if max_ >= 2:
            # [0, 255]
            origin_scale = 255
        elif max_ < 2:
            if min_ > -0.5:
                # [0, 1]
                origin_scale = 1
            elif min_ > -2:
                # [-1, 1]
                origin_scale = 2
            else:
                print("!> Error in scale (%f, %f)" % (max_, min_))

        if kind == 'tanh':
            if origin_scale == 2:
                return x.astype("float32")
            return x.astype("float32") * 2 / origin_scale - 1
        elif kind == 'sigmoid':
            return x.astype("float32") * 2 / origin_scale - 1
        elif kind == 'linear':
            return x.astype("float32") * 255 / origin_scale
        elif kind == 'none':
            return x.astype("float32")

    return func

def get_inverse_process_fn(kind='none'):
    """
    Assume -1 ~ 1, get functions for inverse post processing to 0~255
    """
    # process to 0~255
    if kind == 'tanh':
        return lambda x : ((x + 1) * 127.5).astype("uint8")
    elif kind == 'sigmoid':
        return lambda x : (x * 255.).astype("uint8")
    elif kind == 'linear':
        return lambda x : (x + 127.5).astype("uint8")
    elif kind == 'none':
        return lambda x : x.astype("uint8")

def get_random(kind, shape):
    """
    Unified interface to getting random number:
    this is for easy comparison of different noises
    """
    def random_truncate_normal(shape, max=1.0, mean=0.0):
        res = np.random.normal(loc=mean, scale=max, size=shape)
        res[res > max] = max
        res[res < -max] = -max
        return res

    def random_onehot(shape):
        tmp = np.random.rand(*shape)
        res = np.zeros(shape)
        res[range(shape[0]), tmp.argmax(1)] = 1
        del tmp
        return res

    def random_getchu_restricted(shape, boolean=False):
        """
        Return a generated probability
        """
        res = np.random.rand(*shape)
        stList = [0, 13, 15, 18, 19, 20, 21, 22, 23, 24, 34]
        res = []
        for j in range(shape[0]):
            vec = np.random.rand(shape[1])
            for i in range(len(stList) - 1):
                if stList[i+1] - stList[i] <= 1:
                    # single attribute is rarely set in original dataset, so here the probability also need to be lowered
                    vec[stList[i]] = vec[stList[i]] ** 2
                    if boolean:
                        vec[stList[i]] = float(vec[stList[i]] > 0.5)
                else:
                    if boolean:
                        ind = vec[stList[i]: stList[i+1]].argmax()
                        vec[stList[i]: stList[i+1]] = 0
                        vec[stList[i] + ind] = 1
                    else:
                        # normalize to probability 1
                        vec[stList[i]: stList[i+1]] /= vec[stList[i]: stList[i+1]].sum()
        
        """
        for i in range(len(stList)-1):
            # skip normalization of switch variable
            if stList[i+1] - stList[i] <= 1: continue
            for j in range(shape[0]):
                res[j, stList[i]: stList[i+1]] /= res[j, stList[i]: stList[i+1]].sum()
        """
        return np.stack(res, 0)

    if kind == "uniform":
        return np.random.uniform(-1, 1, shape)
    elif kind == "normal":
        return np.random.normal(loc=0.0, scale=2.0, size=shape).astype("float32")
    elif kind == "boolean":
        return np.random.uniform(0, 1, shape).round()
    elif kind == "onehot":
        return random_onehot(shape)
    elif kind == "getchu_continuous":
        return random_getchu_restricted(shape)
    elif kind == "getchu_boolean":
        return random_getchu_restricted(shape, True)
    elif kind == "truncate_normal":
        return random_truncate_normal(shape)
    else:
        print("!> Getting random method failed")
        return random_uniform(shape)

def fn_seq(func_seq):
    def proc_(x):
        p = x
        for f in func_seq:
            p = f(p)
        return p
    return proc_

def get_resize_image(target_shape):
    def resize_image_(x):
        if type(target_shape) is float:
            x = T.rescale(x, target_shape)

        elif x.shape[0] != target_shape[0] or x.shape[1] != target_shape[1]:
            x = T.resize(x, target_shape)

        return x
    return resize_image_

def get_disturb_image(kind='none'):
    def disturb_image_(x):
        dev = 1
        if kind == 'tanh':
            dev = 0.04
        elif kind == 'sigmoid':
            dev = 0.04
        elif kind == 'linear':
            dev = 4
        elif kind == 'none':
            dev = 4
        x = x + np.random.normal(loc=0.0, scale=dev, size=x.shape)
        x [x < 0] = 0
        x [x > 255] = 255
        return x
    return disturb_image_

def get_flip_image():
    def flip_image_(x):
        return flip_axis(x, axis=1, is_random=True)
    return flip_image_

def get_illumination_disturb():
    def illumination_disturb_(x):
        return illumination(x, gamma=(0.9,1.1), contrast=(0.9,1.1), saturation=(0.9,1.1), is_random=True)
    return illumination_disturb_


##### TensorFlow Operations ######
##################################

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)

def make_losslist_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        if type(v) is tf.Tensor:
            # tensor: split averagely to number of gpu
            in_splits[k] = tf.split(v, num_gpus)
        elif type(v) is list:
            # list of tensors: split each
            gpu_list = []
            for item in v:
                gpu_list.append(tf.split(item, num_gpus))
            in_splits[k] = []
            for i in range(num_gpus):
                in_splits[k].append([gpu_list[j][i] for j in range(len(gpu_list))])
        else:
            # normal arguments
            in_splits[k] = [v] * num_gpus

    out_losses = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            #with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            in_kwargs = {k : v[i] for k, v in in_splits.items()}
            #print(in_kwargs)
            loss_list = fn(**in_kwargs)
            for i in range(len(loss_list)):
                if len(out_losses) <= i:
                    out_losses.append([])
                out_losses[i].append(loss_list[i])

    return [tf.add_n(loss_gpu) for loss_gpu in out_losses]

def debug_tensor(x, msg=None):
    """
    Print max, min value of an tensor
    """
    return tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], msg)

def get_grid_image_summary(img_tensor, n):
    """
    rearrange a 4-D image tensor to show an n x n grid.
    Args:
        img_tensor : TF tensor, 4-D
        n           : grid size. Make sure batch size is larger than n*n
    """
    if n == 1: return img_tensor[:1]

    img_list = []
    show_img_array = img_tensor[:n**2]
    for i in range(n):
        tmp_list = []
        for j in range(n):
            tmp_list.append(show_img_array[i*n+j:i*n+j+1])
        img_list.append(tf.concat(tmp_list, axis=1))
    return tf.concat(img_list, axis=2)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection="spectral_norm_update_ops", with_sigma=False):
    def _l2normalize(v, eps=1e-4):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    # [in_dim, out_dim]
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    
    _, u_final, v_final = power_iteration(
        tf.constant(0, dtype=tf.int32),
        u,
        tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    
    if update_collection is None:
        # print('=> Setting update_collection to None will make u being updated every W execution. This maybe undesirable. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != "no_ops":
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar

