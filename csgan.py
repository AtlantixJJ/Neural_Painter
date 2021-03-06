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

from lib.ptseg import *

BEST_MODEL = "success/goodmodel_dragan_anime1"

# ------ control ----- #

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to debug")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_boolean("resume", False, "If to resume from previous training. Incompatible with --resume.")
tf.app.flags.DEFINE_integer("save_iter", 20000, "saving interval")
tf.app.flags.DEFINE_integer("num_worker", 1, "threads")
tf.app.flags.DEFINE_string("train_dir", "logs/simple_getchu", "log dir")

# ----- model type ------ #

tf.app.flags.DEFINE_boolean("cgan", False, "If use ACGAN")
tf.app.flags.DEFINE_integer("img_size", 128, "64 | 128")
tf.app.flags.DEFINE_string("model_name", "hg", "model type: simple | simple_mask | hg | hg_mask")
tf.app.flags.DEFINE_string("data_dir", "/home/atlantix/data/celeba/img_align_celeba.zip", "data path")
tf.app.flags.DEFINE_boolean("cbn_project", False, "If to project to depth dim")

# ------ train control ----- #

tf.app.flags.DEFINE_boolean("use_cache", False, "If use cache to prevent cactastrophic forgetting.")
tf.app.flags.DEFINE_integer("gpu", 4, "which gpu to use")
tf.app.flags.DEFINE_float("g_lr", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("d_lr", 4e-4, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "training batch size")
tf.app.flags.DEFINE_integer("num_iter", 200000, "training iteration")
tf.app.flags.DEFINE_integer("dec_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("disc_iter", 5, "discriminator training iter")
tf.app.flags.DEFINE_integer("gen_iter", 1, "generative training iter")

FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

NUM_WORKER = FLAGS.num_worker
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    size = FLAGS.img_size

    if FLAGS.cgan:
        npy_dir = FLAGS.data_dir.replace(".zip", "") + '.npy'
    else:
        npy_dir = None

    if "celeb" in FLAGS.data_dir:
        dataset = dataloader.CelebADataset(FLAGS.data_dir,
            img_size=(size, size),
            npy_dir=npy_dir)
    elif "cityscapes" in FLAGS.data_dir:
        augmentations = Compose([RandomCrop(size * 4), Scale(size * 2), RandomRotate(10), RandomHorizontallyFlip(), RandomSizedCrop(size)])
        dataset = dataloader.cityscapesLoader(FLAGS.data_dir,
            is_transform=True,
            augmentations=augmentations,
            img_size=(size, size))
    else:
        dataset = dataloader.FileDataset(FLAGS.data_dir,
            npy_dir=npy_dir,
            img_size=(size, size),
            shuffle=True)

    dl = DataLoader(dataset, batch_size=FLAGS.batch_size // 64, shuffle=True, num_workers=NUM_WORKER, collate_fn=dataloader.default_collate)

    # TF Input
    x_fake_sample = tf.placeholder(tf.float32, [None, size, size, 3], name="x_fake_sample")
    x_real = tf.placeholder(tf.float32, [None, size, size, 3], name="x_real")
    s_real = tf.placeholder(tf.float32, [None, size, size, 3], name='s_real')
    z_noise = tf.placeholder(tf.float32, [None, 128], name="z_noise")

    if FLAGS.cgan:
        c_noise = tf.placeholder(tf.float32, [None, dataset.class_num], name="c_noise")
        c_label = tf.placeholder(tf.float32, [None, dataset.class_num], name="c_label")
        gen_input = [z_noise, c_noise]
    else:
        gen_input = z_noise

    gen_model, disc_model = getattr(config, FLAGS.model_name)(FLAGS.img_size, dataset.class_num)
    gen_model.cbn_project = FLAGS.cbn_project
    
    x_fake = gen_model(gen_input, update_collection=None)
    gen_model.set_reuse()
    gen_model.x_fake = x_fake

    disc_real, real_cls_logits = disc_model(x_real, update_collection=None)
    disc_model.set_reuse()
    disc_fake, fake_cls_logits = disc_model(x_fake, update_collection="no_ops")
    disc_model.disc_real        = disc_real       
    disc_model.disc_fake        = disc_fake       
    disc_model.real_cls_logits = real_cls_logits
    disc_model.fake_cls_logits = fake_cls_logits

    int_sum_op = []
    
    if FLAGS.use_cache:
        disc_fake_sample = disc_model(x_fake_sample)[0]
        disc_cost_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake_sample,
                labels=tf.zeros_like(disc_fake_sample)), name="cost_disc_fake_sample")
        disc_cost_sample_sum = tf.summary.scalar("disc_sample", disc_cost_sample)

        fake_sample_grid = ops.get_grid_image_summary(x_fake_sample, 4)
        int_sum_op.append(tf.summary.image("fake sample", fake_sample_grid))

        sample_method = [disc_cost_sample, disc_cost_sample_sum, x_fake_sample]
    else:
        sample_method = None

    grid_x_fake = ops.get_grid_image_summary(gen_model.x_fake, 4)
    int_sum_op.append(tf.summary.image("generated image", grid_x_fake))

    grid_x_real = ops.get_grid_image_summary(x_real, 4)
    int_sum_op.append(tf.summary.image("real image", grid_x_real))

    if FLAGS.cgan:
        loss.classifier_loss(gen_model, disc_model, x_real, c_label, c_noise,
        weight=1.0)

    loss.hinge_loss(gen_model, disc_model, adv_weight=1.0)
    
    int_sum_op = tf.summary.merge(int_sum_op)

    ModelTrainer = trainer.base_gantrainer.BaseGANTrainer(
        int_sum_op=int_sum_op,
        dataloader=dl,
        FLAGS=FLAGS,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_input=gen_input,
        x_real=x_real,
        label=c_label,
        sample_method=sample_method)

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    print("=> Build train op")
    ModelTrainer.build_train_op()
    
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
