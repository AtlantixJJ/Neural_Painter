import matplotlib
matplotlib.use("agg")
import tensorflow as tf
import os
from PIL import Image
import config, model, loss, trainer
from lib import ops
import skimage.io as io
import numpy as np

tf.app.flags.DEFINE_string("rpath", "./gen.npz", "The path to npz param")
tf.app.flags.DEFINE_integer("img_size", 64, "64 | 128")
tf.app.flags.DEFINE_integer("gpu", 2, "which gpu to use")
tf.app.flags.DEFINE_boolean("cgan", True, "If use ACGAN")
tf.app.flags.DEFINE_integer("class_num", 40, "34 | 40")
tf.app.flags.DEFINE_string("model_name", "simple_mask", "model type: simple | simple_mask | hg | hg_mask")

FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

size = FLAGS.img_size

# TF Input
z_noise = tf.placeholder(tf.float32, [None, 128], name="z_noise")

if FLAGS.cgan:
    c_noise = tf.placeholder(tf.float32, [None, FLAGS.class_num], name="c_noise")
    c_label = tf.placeholder(tf.float32, [None, FLAGS.class_num], name="c_label")
    gen_input = [z_noise, c_noise]
else:
    gen_input = z_noise

gen_model, disc_model = getattr(config, FLAGS.model_name)(FLAGS.img_size, FLAGS.class_num)
x_fake = gen_model(gen_input)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run([tf.global_variables_initializer()])
gen_model.load_from_npz(FLAGS.rpath, sess)

feed_dict = {
    gen_model.keep_prob : 1.0,
    gen_model.training: False
}

def random_gen(save=True):
    feed_dict.update({
        z_noise: ops.random_normal((1, 128)),
        c_noise: ops.random_boolean_by_shape((1, FLAGS.class_num))
    })

    x_fake_sample, masks, outs = sess.run([x_fake, gen_model.masks, gen_model.outs], feed_dict)
    masks = ops.get_grid_image(np.array(masks)[:, 0], 4)
    outs = np.tanh(ops.get_grid_image(np.array(outs)[:, 0], 4))
    outs[outs<-1] = -1
    outs[outs>1] = 1

    if save:
        io.imsave("x_fake_sample.png", x_fake_sample[0])
        io.imsave("x_outs.png", outs[0])
        io.imsave("x_masks.png", masks[0])

    return x_fake_sample[0], masks[0], outs[0]

random_gen()

