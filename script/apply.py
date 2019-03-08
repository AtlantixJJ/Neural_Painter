import sys
sys.path.insert(0, ".")
import tensorflow as tf

import numpy as np
import time
import pprint
import os

from lib import files, ops, utils
import model
import loss
import config as cfg

BEST_MODEL = "success/goodmodel_fulldata_dragan/goodmodel_dragan_anime1/"

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_integer("mode", 2, "1 for ckpt; 2 for weight")
tf.app.flags.DEFINE_boolean("cgan", False, "If use ACGAN")
tf.app.flags.DEFINE_string("model_name", "goodmodel_dragan_anime1", "model type: simple_getchu")
tf.app.flags.DEFINE_integer("gpu", 6, "which gpu to use")
FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def apply_from_ckpt():
    model_dir = "success/goodmodel_fulldata_dragan/goodmodel_dragan_anime1/"
    model_name = "goodmodel_dragan_anime1-00200000"
    model_path = model_dir + model_name
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(model_path + ".meta")
    saver.restore(sess, model_path)

    gen_output = sess.graph.get_tensor_by_name('GoodGenerator/gen_out:0')
    z_noise = sess.graph.get_tensor_by_name('z_noise:0')
    c_noise = sess.graph.get_tensor_by_name('c_noise:0')
    training = sess.graph.get_tensor_by_name("training:0")
    training_1 = sess.graph.get_tensor_by_name("training_1:0")

    feed_dict = {
        z_noise: ops.random_truncate_normal((16, 128), 1, 0),
        c_noise: ops.random_boolean((16, 34), True),
        training: False,
        training_1: False
    }

    gen_img = (sess.run([gen_output], feed_dict)[0] + 1) / 2
    utils.save_batch_img(gen_img, "ex_ckpt.png", 4)

    return gen_output, feed_dict, sess

def apply_from_weight():
    # get configuration
    print("=> Get configuration")
    TFLAGS = cfg.get_train_config(FLAGS.model_name)
    gen_model, gen_config, disc_model, disc_config = model.get_model(FLAGS.model_name, TFLAGS)

    z_noise = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="z_noise")
    x_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")

    if FLAGS.cgan:
        c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
        c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # build model
    with tf.variable_scope(gen_model.field_name):
        if FLAGS.cgan:
            gen_model.is_cgan = True
            x_fake = gen_model.build_inference([z_noise, c_noise])
        else:
            x_fake = gen_model.build_inference(z_noise)

    with tf.variable_scope(disc_model.field_name):
        if FLAGS.cgan:
            disc_model.is_cgan = True
            disc_model.disc_real_out = disc_model.build_inference([x_fake, c_label])[0]
        else:
            disc_model.disc_real_out = disc_model.build_inference(x_fake)[0]

    gen_vars = files.load_npz(BEST_MODEL, "goodmodel_dragan_anime1_gen.npz")
    disc_vars = files.load_npz(BEST_MODEL, "goodmodel_dragan_anime1_disc.npz")

    feed_dict = {
        z_noise: ops.random_truncate_normal((16, 128), 1, 0),
        c_noise: ops.random_boolean((16, 34), True),
        gen_model.training: False,
        disc_model.training: False,
        gen_model.keep_prob: 1.0,
        disc_model.keep_prob: 1.0
    }

    gen_model.get_trainable_variables()
    disc_model.get_trainable_variables()
    gen_moving_vars = [v for v in tf.global_variables() if v.name.find("Gen") > -1 and v.name.find("moving") > -1]
    disc_moving_vars = [v for v in tf.global_variables() if v.name.find("Disc") > -1 and v.name.find("moving") > -1]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run([tf.global_variables_initializer()])

    gen_img = (sess.run([x_fake], feed_dict)[0] + 1) / 2
    utils.save_batch_img(gen_img, "ex_rand.png", 4)

    files.assign_params(sess, gen_vars, gen_model.vars + gen_moving_vars)
    files.assign_params(sess, disc_vars, disc_model.vars + disc_moving_vars)

    gen_img = (sess.run([x_fake], feed_dict)[0] + 1) / 2
    utils.save_batch_img(gen_img, "ex_load.png", 4)

    return x_fake, feed_dict, sess

if __name__ == "__main__":
    if FLAGS.mode == 1:
        gen_out, feed_dict, sess = apply_from_ckpt()
    elif FLAGS.mode == 2:
        gen_out, feed_dict, sess = apply_from_weight()