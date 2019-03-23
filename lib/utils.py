"""
Various utility.
"""
import numpy as np
import os
import tensorflow as tf

import lib.ops as ops
import skimage.io as io

def find_tensor_by_name(name, tensorlist):
    name = name[name.find("/")+1:]
    for i, t in enumerate(tensorlist):
        tname = t.name[t.name.find("/")+1:]
        if name == tname:
            return t

def partial(func, **kwargs):
    def _func(*kargs):
        return func(*kargs, **kwargs)
    return _func

def linear_interpolate(x_bg, x_ed, y_bg, y_ed, cur_x):
    """
    Scalar only.
    Args:
    x_bg    :   Line starts from x_bg
    x_ed    :   Line ends at x_ed
    y_bg    :   Value starts from y_bg
    y_ed    :   Value ends at y_ed
    cur_x   :   Currently x is at cur_x
    """
    k = float(y_ed - y_bg) / float(x_ed - x_bg)
    est_y = y_bg + k * (cur_x - x_bg)
    return est_y

def get_interval_summary(data1, data2, batch_size):
    len_edge = int(np.sqrt(batch_size))
    if len_edge > 4:
        len_edge = 4
    generated_samples = ops.get_grid_image_summary(data1, len_edge)
    real_samples = ops.get_grid_image_summary(data2, len_edge)
    gen_image_sum = tf.summary.image("generated", generated_samples + 1)
    real_image_sum = tf.summary.image("real", real_samples + 1)
    sum_interval_op = tf.summary.merge([gen_image_sum, real_image_sum])
    return sum_interval_op

def save_batch_img(img, name, n=4):
    """
    Args:
    img :   (b, s, s, c)
    """
    img = ops.get_grid_image(img, n) * 255.
    img = img[0].astype("uint8")
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    io.imsave(name, img)

def stripe_digits(str):
    n = []
    for c in str:
        if not c.isdigit():
            n.append(c)
    return ''.join(n)

def stripe_dataset(str):
    str = stripe_digits(str)
    dataset_name = str.split("_")[-2]
    return str.replace("_" + dataset_name, "")

def stripe_loss(str):
    str = stripe_digits(str)
    loss_name = str.split("_")[1]
    return str.replace("_" + loss_name, "")

def shuffle(X, shuffle_parts):
    chunk_size = len(X) // shuffle_parts
    shuffled_range = np.array(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer

    return X

def load_mnist_4d(data_dir):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 1, 28, 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 1, 28, 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    
    return trX, teX, trY, teY