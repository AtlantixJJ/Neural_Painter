import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def linear(name, input, output_dim, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("weight", shape=[input.get_shape()[-1], output_dim], initializer=tf.orthogonal_initializer)
        b = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(.0))

        x = tf.matmul(input, w) + b
        with tf.device("/device:CPU:0"):
            x = tf.Print(x, [tf.reduce_sum(tf.abs(w)), tf.reduce_sum(tf.abs(b))], name + "/weight_bias: ")
    
    return x

def tower(x, reuse=False):
    rec_tensor = []
    rec_name = []
    x = linear("fc1", x, 10, reuse)
    rec_tensor.append(x); rec_name.append("fc1")
    x = linear("fc2", x, 1, reuse)
    rec_tensor.append(x); rec_name.append("fc2")
    return x, rec_tensor, rec_name

x = tf.placeholder(tf.float32, [None, 3])
x_data = np.random.rand(5, 3)
feed_dict = {x: x_data}
ys = []
rec_xs = []
rec_names = []

for i in range(2):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
        y_, v_, n_ = tower(x, i>0)
        ys.append(y_)
        rec_xs.append(v_)
        rec_names.append(n_)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

print("=> Check forward")
for i in range(2):
    print("=> Check GPU %d" % i)
    for j in range(len(rec_xs[i])):
        t = sess.run(rec_xs[i][j], feed_dict)[0]
        l1norm = np.sum(np.abs(t))
        print("=> %s: %.5f" % (rec_names[i][j], l1norm))

    