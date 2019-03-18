"""
GAN Trainer Family.
Common options:
--model_name simple_ian_mnist1 : [network structure name]_[loss family name]_[dataset name]_[version number]
--cgan  : if to use labels in GAN (that is ACGAN)
"""
import matplotlib
matplotlib.use("agg")
import tensorflow as tf

import time, pprint, os
import numpy as np
from scipy import misc

from torch.utils.data import DataLoader

# model
import config, model, loss, trainer
from lib import utils, dataloader, ops

BEST_MODEL = "success/goodmodel_dragan_anime1"
NUM_WORKER = 1
# ------ control ----- #

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to debug")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_boolean("resume", False, "If to resume from previous training. Incompatible with --resume.")
tf.app.flags.DEFINE_integer("save_iter", 20000, "saving interval")

tf.app.flags.DEFINE_string("train_dir", "logs/simple_getchu", "log dir")

# ----- model type ------ #

tf.app.flags.DEFINE_boolean("cgan", False, "If use ACGAN")
tf.app.flags.DEFINE_integer("img_size", 64, "64 | 128")
tf.app.flags.DEFINE_string("model_name", "hg", "model type: simple | simple_mask | hg | hg_mask")
tf.app.flags.DEFINE_string("data_dir", "/home/atlantix/data/getchu/true_face.zip", "data path")

# ------ train control ----- #

tf.app.flags.DEFINE_boolean("use_cache", False, "If use cache to prevent cactastrophic forgetting.")
tf.app.flags.DEFINE_string("gpu", "0,1,5,6", "which gpu to use")
tf.app.flags.DEFINE_float("g_lr", 5e-5, "learning rate")
tf.app.flags.DEFINE_float("d_lr", 2e-4, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "training batch size")
tf.app.flags.DEFINE_integer("num_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("dec_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("disc_iter", 2, "discriminator training iter")
tf.app.flags.DEFINE_integer("gen_iter", 1, "generative training iter")

FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

def main():
    size = FLAGS.img_size
    num_gpus = len(FLAGS.gpu.split(","))

    if FLAGS.cgan:
        npy_dir = FLAGS.data_dir.replace(".zip", "") + '.npy'
    else:
        npy_dir = None

    if "celeb" in FLAGS.data_dir:
        dataset = dataloader.CelebADataset(FLAGS.data_dir,
            img_size=(size, size),
            npy_dir=npy_dir)
    else:
        dataset = dataloader.FileDataset(FLAGS.data_dir,
            npy_dir=npy_dir,
            img_size=(size, size),
            shuffle=True)
    dl = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=NUM_WORKER)

    # TF Input
    x_fake_sample = tf.placeholder(tf.float32, [FLAGS.batch_size, size, size, 3], name="x_fake_sample")
    x_real = tf.placeholder(tf.float32, [FLAGS.batch_size, size, size, 3], name="x_real")
    s_real = tf.placeholder(tf.float32, [FLAGS.batch_size, size, size, 3], name='s_real')
    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, 128], name="z_noise")

    if FLAGS.cgan:
        c_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, dataset.class_num], name="c_noise")
        c_label = tf.placeholder(tf.float32, [FLAGS.batch_size, dataset.class_num], name="c_label")
        gen_input = [z_noise, c_noise]
    else:
        gen_input = z_noise

    gen_model, disc_model = getattr(config, FLAGS.model_name)(FLAGS.img_size, dataset.class_num)

    ModelTrainer = trainer.base_gantrainer.BaseGANTrainer(
        int_sum_op=[],
        dataloader=dl,
        FLAGS=FLAGS,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_input=gen_input,
        x_real=x_real,
        label=c_label,
        sample_method=None)

    g_optim = tf.train.AdamOptimizer(
                learning_rate=ModelTrainer.g_lr,
                beta1=0.,
                beta2=0.999)
    d_optim = tf.train.AdamOptimizer(
                learning_rate=ModelTrainer.d_lr,
                beta1=0.,
                beta2=0.999)

    g_tower_grads = []
    d_tower_grads = []

    """
    with tf.device("/device:GPU:0"):
        x_fake = gen_model(gen_input, update_collection="no_ops")
        gen_model.set_reuse()
        disc_real, real_cls_logits = disc_model(x_real, update_collection="no_ops")
        disc_model.set_reuse()
        disc_fake, fake_cls_logits = disc_model(x_fake, update_collection="no_ops")      
        params = tf.trainable_variables()
        gen_model.vars = [i for i in params if gen_model.name in i.name]
        disc_model.vars = [i for i in params if disc_model.name in i.name]
    """

    def tower(gen_input, x_real, c_label=None, c_noise=None, update_collection=None):
        x_fake = gen_model(gen_input, update_collection=update_collection)
        gen_model.set_reuse()
        gen_model.x_fake = x_fake

        disc_real, real_cls_logits = disc_model(x_real, update_collection=update_collection)
        disc_model.set_reuse()
        disc_fake, fake_cls_logits = disc_model(x_fake, update_collection=update_collection)
        disc_model.disc_real        = disc_real       
        disc_model.disc_fake        = disc_fake       
        disc_model.real_cls_logits = real_cls_logits
        disc_model.fake_cls_logits = fake_cls_logits

        if FLAGS.cgan:
            loss.classifier_loss(gen_model, disc_model, x_real, c_label, c_noise,
            weight=1.0)

        loss.hinge_loss(gen_model, disc_model, adv_weight=1.0)

        params = tf.trainable_variables()
        gen_model.vars = [i for i in params if gen_model.name in i.name]
        disc_model.vars = [i for i in params if disc_model.name in i.name]

        g_tower_grads.append(g_optim.compute_gradients(gen_model.cost, var_list=gen_model.vars, colocate_gradients_with_ops=True))
        d_tower_grads.append(d_optim.compute_gradients(disc_model.cost, var_list=disc_model.vars, colocate_gradients_with_ops=True))

    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            if grad_and_vars[0][0] is None:
                print(grad_and_vars[0][1])
                continue
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    sbs = FLAGS.batch_size // num_gpus

    for i in range(num_gpus):
        gen_model.cost = disc_model.cost = 0
        if i == 0:
            update_collection = None
        else:
            update_collection = "no_ops"

        with tf.device("/device:GPU:%d" % i):
            if FLAGS.cgan:
                tower(
                    [z_noise[sbs*i:sbs*(i+1)], c_noise[sbs*i:sbs*(i+1)]],
                    x_real[sbs*i:sbs*(i+1)],
                    c_label[sbs*i:sbs*(i+1)],
                    c_noise[sbs*i:sbs*(i+1)], update_collection=update_collection)
            else:
                tower(z_noise[sbs*i:sbs*(i+1)],x_real[sbs*i:sbs*(i+1)], update_collection=update_collection)

        if i == 0:
            int_sum_op = []

            grid_x_fake = ops.get_grid_image_summary(gen_model.x_fake, 4)
            int_sum_op.append(tf.summary.image("generated image", grid_x_fake))

            grid_x_real = ops.get_grid_image_summary(x_real, 4)
            int_sum_op.append(tf.summary.image("real image", grid_x_real))

            int_sum_op = tf.summary.merge(int_sum_op)

    with tf.device("/device:GPU:0"):
        g_grads = average_gradients(g_tower_grads)
        d_grads = average_gradients(d_tower_grads)

    gen_model.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=gen_model.name + "/")
    disc_model.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=disc_model.name + "/")
    
    with tf.device("/device:GPU:0"):
        with tf.control_dependencies(gen_model.update_ops):
                gen_model.train_op = g_optim.apply_gradients(g_grads)
        with tf.control_dependencies(disc_model.update_ops):
                disc_model.train_op = d_optim.apply_gradients(d_grads)

    gen_model.sum_op = tf.summary.merge(gen_model.sum_op)
    disc_model.sum_op = tf.summary.merge(disc_model.sum_op)
    
    ModelTrainer.int_sum_op = int_sum_op

    print("=> ##### Generator Variable #####")
    gen_model.print_trainble_vairables()
    print("=> ##### Discriminator Variable #####")
    disc_model.print_trainble_vairables()
    print("=> ##### All Variable #####")
    for v in tf.trainable_variables():
        print("%s\t\t\t\t%s" % (v.name, str(v.get_shape().as_list())))
    print("=> #### Moving Variable ####")
    for v in tf.global_variables():
        if "moving" in v.name:
            print("%s\t\t\t\t%s" % (v.name, str(v.get_shape().as_list())))


    ModelTrainer.init_training()
    ModelTrainer.train()

if __name__ == "__main__":
    main()
