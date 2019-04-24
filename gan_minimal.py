"""
GAN Trainer Family.
example:
python gan_minimal.py --gpu 0 --model_name sample --cgan
"""
import matplotlib
matplotlib.use("agg") # normal setting
import tensorflow as tf
import time, pprint, os
import numpy as np
from scipy import misc

# modules in current project
import config, model, loss, trainer
from lib import utils, dataloader, ops
# pytorch segmentation related utils
from lib.ptseg import *

# ------ control flags ----- #

tf.app.flags.DEFINE_string("rpath", "./", "The path to npz param, for reloading")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to show debug messages")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_boolean("resume", False, "If to resume from previous training. Incompatible with --resume.")
tf.app.flags.DEFINE_integer("save_iter", 20000, "saving iteration interval")
tf.app.flags.DEFINE_integer("num_worker", 1, "threads")
tf.app.flags.DEFINE_string("train_dir", "", "log dir")

# ----- model type flags ------ #

tf.app.flags.DEFINE_boolean("cgan", True, "If to use ACGAN")
tf.app.flags.DEFINE_integer("img_size", 128, "The size of input image, 64 | 128")
tf.app.flags.DEFINE_string("model_name", "simple", "model type: simple | simple_mask | hg | hg_mask")
tf.app.flags.DEFINE_string("data_dir", "/home/atlantix/data/celeba/img_align_celeba.zip", "data path")

# ------ train control flags ----- #

tf.app.flags.DEFINE_boolean("use_cache", False, "If to use cache to prevent cactastrophic forgetting.")
tf.app.flags.DEFINE_integer("gpu", 4, "which gpu to use")
tf.app.flags.DEFINE_float("g_lr", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("d_lr", 4e-4, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "training batch size")
tf.app.flags.DEFINE_integer("num_iter", 200000, "training iteration")
tf.app.flags.DEFINE_integer("dec_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("disc_iter", 1, "discriminator training iter")
tf.app.flags.DEFINE_integer("gen_iter", 1, "generative training iter")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    size = FLAGS.img_size

    if FLAGS.cgan:
        # the label file should be npy format
        npy_dir = FLAGS.data_dir.replace(".zip", "") + '.npy'
    else:
        npy_dir = None

    if "celeb" in FLAGS.data_dir:
        dataset = dataloader.CelebADataset(FLAGS.data_dir,
            img_size=(size, size),
            npy_dir=npy_dir)
    elif "cityscapes" in FLAGS.data_dir:
        # outdated
        augmentations = Compose([RandomCrop(size * 4), Scale(size * 2), RandomRotate(10), RandomHorizontallyFlip(), RandomSizedCrop(size)])
        dataset = dataloader.cityscapesLoader(FLAGS.data_dir,
            is_transform=True,
            augmentations=augmentations,
            img_size=(size, size))
        FLAGS.batch_size /= 64
    else:
        dataset = dataloader.FileDataset(FLAGS.data_dir,
            npy_dir=npy_dir,
            img_size=(size, size))

    dl = dataloader.TFDataloader(dataset, FLAGS.batch_size, dataset.file_num // FLAGS.batch_size)
    
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
        c_label = c_noise = None

    # look up the config function from lib.config module
    gen_model, disc_model = getattr(config, FLAGS.model_name)(FLAGS.img_size, dataset.class_num)
    
    gen_model.label = c_noise
    x_fake = gen_model(gen_input)
    gen_model.set_reuse()
    gen_model.x_fake = x_fake

    disc_model.label = c_noise
    disc_fake, fake_cls_logits = disc_model(x_fake)
    disc_model.set_reuse()

    disc_model.label = c_label
    disc_real, real_cls_logits = disc_model(x_real)
    disc_model.disc_real        = disc_real
    disc_model.disc_fake        = disc_fake 
    disc_model.real_cls_logits = real_cls_logits
    disc_model.fake_cls_logits = fake_cls_logits

    raw_gen_cost, raw_disc_real, raw_disc_fake = loss.hinge_loss(gen_model, disc_model, adv_weight=1.0, summary=False)
    disc_model.disc_real_loss = raw_disc_real
    disc_model.disc_fake_loss = raw_disc_fake

    if FLAGS.cgan:
        real_cls_cost, fake_cls_cost = loss.classifier_loss(gen_model, disc_model, x_real, c_label, c_noise,
        weight=1.0 / dataset.class_num, summary=False)
        subloss_names = ["fake_cls", "real_cls", "gen", "disc_real", "disc_fake"]
        sublosses = [fake_cls_cost, real_cls_cost, raw_gen_cost, raw_disc_real, raw_disc_fake]
    else:
        subloss_names =  ["gen", "disc_real", "disc_fake"]
        sublosses = [raw_gen_cost, raw_disc_real, raw_disc_fake]

    step_sum_op = [] # summary at every step

    for n,l in zip(subloss_names, sublosses):
        step_sum_op.append(tf.summary.scalar(n, l))
    if gen_model.debug or disc_model.debug:
        for model_var in tf.global_variables():
            if gen_model.name in model_var.op.name or disc_model.name in model_var.op.name:
                step_sum_op.append(tf.summary.histogram(model_var.op.name, model_var))
    step_sum_op = tf.summary.merge(step_sum_op)

    int_sum_op = [] # summary at some interval

    grid_x_fake = ops.get_grid_image_summary(gen_model.x_fake, 4)
    int_sum_op.append(tf.summary.image("generated image", grid_x_fake))

    grid_x_real = ops.get_grid_image_summary(x_real, 4)
    int_sum_op.append(tf.summary.image("real image", grid_x_real))

    int_sum_op = tf.summary.merge(int_sum_op)

    ModelTrainer = trainer.base_gantrainer.BaseGANTrainer(
        int_sum_op=int_sum_op,
        step_sum_op=step_sum_op,
        dataloader=dl,
        FLAGS=FLAGS,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_input=gen_input,
        x_real=x_real,
        label=c_label)

    print("=> Build train op")
    ModelTrainer.build_train_op()
    
    print("=> ##### Generator Variable #####")
    gen_model.print_trainble_vairables()
    print("=> ##### Discriminator Variable #####")
    disc_model.print_trainble_vairables()
    print("=> #### Moving Variable ####")
    for v in tf.global_variables():
        if "moving" in v.name:
            print("%s\t\t\t\t%s" % (v.name, str(v.get_shape().as_list())))
    print("=> #### Generator update dependency ####")
    for v in gen_model.update_ops:
        print("%s" % (v.name))
    print("=> #### Discriminator update dependency ####")
    for v in disc_model.update_ops:
        print("%s" % (v.name))
    ModelTrainer.init_training()
    ModelTrainer.train()

if __name__ == "__main__":
    main()
