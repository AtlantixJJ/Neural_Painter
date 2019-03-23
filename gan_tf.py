"""
GAN Trainer Family.
Common options:
--model_name simple_ian_mnist1 : [network structure name]_[loss family name]_[dataset name]_[version number]
--cgan  : if to use labels in GAN (that is ACGAN)
"""
import matplotlib
matplotlib.use("agg") # normal setting
import tensorflow as tf
import time, pprint, os
import numpy as np
from scipy import misc

# Use the dataloader of pytorch
from torch.utils.data import DataLoader

# modules in current project
import config, model, loss, trainer
from lib import utils, dataloader, ops
# pytorch segmentation related utils
from lib.ptseg import *

# The save dir of best model
BEST_MODEL = "success/goodmodel_dragan_anime1"

# ------ control flags ----- #

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param, for reloading")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to show debug messages")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_boolean("resume", False, "If to resume from previous training. Incompatible with --resume.")
tf.app.flags.DEFINE_integer("save_iter", 20000, "saving iteration interval")
tf.app.flags.DEFINE_integer("num_worker", 1, "threads")
tf.app.flags.DEFINE_string("train_dir", "", "log dir")

# ----- model type flags ------ #

tf.app.flags.DEFINE_boolean("cgan", True, "If to use ACGAN")
tf.app.flags.DEFINE_integer("img_size", 64, "The size of input image, 64 | 128")
tf.app.flags.DEFINE_string("model_name", "hg", "model type: simple | simple_mask | hg | hg_mask")
tf.app.flags.DEFINE_string("data_dir", "/home/atlantix/data/celeba/img_align_celeba.zip", "data path")
tf.app.flags.DEFINE_boolean("cbn_project", True, "If to project to depth dim")
tf.app.flags.DEFINE_integer("bn", 1, "")
tf.app.flags.DEFINE_integer("phases", 1, "1 | 2")

# ------ train control flags ----- #

tf.app.flags.DEFINE_boolean("use_cache", False, "If to use cache to prevent cactastrophic forgetting.")
tf.app.flags.DEFINE_integer("gpu", 4, "which gpu to use")
tf.app.flags.DEFINE_float("g_lr", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("d_lr", 4e-4, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "training batch size")
tf.app.flags.DEFINE_integer("num_iter", 200000, "training iteration")
tf.app.flags.DEFINE_integer("dec_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("disc_iter", 1, "discriminator training iter")
tf.app.flags.DEFINE_integer("gen_iter", 1, "generative training iter")

FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

NUM_WORKER = FLAGS.num_worker
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    size = FLAGS.img_size

    if len(FLAGS.train_dir) < 1:
        bn_name = ["nobn", "caffebn", "simplebn", "defaultbn", "cbn"]
        FLAGS.train_dir = os.path.join("logs", FLAGS.model_name + "_" + bn_name[FLAGS.bn] + "_" + str(FLAGS.phases))

    if FLAGS.cgan:
        # the label file is npy format
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

    # look up the config function from lib.config module
    gen_model, disc_model = getattr(config, FLAGS.model_name)(FLAGS.img_size, dataset.class_num)
    disc_model.norm_mtd = FLAGS.bn

    x_fake = gen_model(gen_input, update_collection=None)
    gen_model.set_reuse()
    gen_model.x_fake = x_fake
    
    disc_model.set_label(c_noise)
    if FLAGS.phases > 1:
        disc_model.set_phase("fake")
    else:
        disc_model.set_phase("default")
    disc_fake, fake_cls_logits = disc_model(x_fake, update_collection=None)
    disc_model.set_reuse()

    """ for debug
    tensors = gen_model.recorded_tensors + disc_model.recorded_tensors
    names = gen_model.recorded_names + disc_model.recorded_names
    grads = tf.gradients(disc_fake, tensors)

    grad_sums = []
    for n,g,t in zip(names, grads, tensors):
        print(n, g)
        if g is not None:
            grad_sums.append(tf.summary.histogram("grad/" + n, g))
    gen_model.sum_op.extend(grad_sums)

    disc_model.recorded_tensors = []
    disc_model.recorded_names = []
    """

    disc_model.set_label(c_label)
    if FLAGS.phases > 1:
        disc_model.set_phase("real")
    else:
        disc_model.set_phase("default")
    disc_real, real_cls_logits = disc_model(x_real, update_collection=None)
    disc_model.disc_real        = disc_real
    disc_model.disc_fake        = disc_fake 
    disc_model.real_cls_logits = real_cls_logits
    disc_model.fake_cls_logits = fake_cls_logits

    """ for debug
    grads = tf.gradients(disc_real, disc_model.recorded_tensors)

    grad_sums = []
    for n,g,t in zip(disc_model.recorded_names, grads, disc_model.recorded_tensors):
        print(n, g)
        if g is not None:
            grad_sums.append(tf.summary.histogram("grad/" + n, g))
    disc_model.sum_op.extend(grad_sums)
    """

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

    int_sum_op = tf.summary.merge(int_sum_op)

    raw_gen_cost, raw_disc_real, raw_disc_fake = loss.hinge_loss(gen_model, disc_model, adv_weight=1.0)#dataset.class_num)
    disc_model.disc_real_loss = raw_disc_real
    disc_model.disc_fake_loss = raw_disc_fake

    #bncloss, sum_ = loss.batchnorm_contrast_loss("fake", "real")
    #gen_model.cost += bncloss
    #gen_model.sum_op.extend(sum_)

    if FLAGS.cgan:
        loss.classifier_loss(gen_model, disc_model, x_real, c_label, c_noise,
        weight=1.0 / dataset.class_num)

    ModelTrainer = trainer.base_gantrainer.BaseGANTrainer(#trainer.separated_gantrainer.SeparatedGANTrainer(#
        int_sum_op=int_sum_op,
        dataloader=dl,
        FLAGS=FLAGS,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_input=gen_input,
        x_real=x_real,
        label=c_label,
        sample_method=sample_method)

    #command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    #command_controller.start_thread()

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
