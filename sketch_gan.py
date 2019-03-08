"""
GAN Trainer Family.
Common options:
--model_name simple_ian_mnist1 : [network structure name]_[loss family name]_[dataset name]_[version number]
--cgan  : if to use labels in GAN (that is ACGAN)
"""
import tensorflow as tf

import time
import pprint
import numpy as np
from scipy import misc
import os

# model
import model
import config as cfg
import loss
import trainer
from lib import utils, dataloader, ops

BEST_MODEL = "success/goodmodel_dragan_anime1"

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to debug")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_string("resume", False, "If to resume from previous expr. Incompatible with --reload.")
tf.app.flags.DEFINE_boolean("cgan", False, "If use ACGAN")
tf.app.flags.DEFINE_string("model_name", "simple_getchu", "model type: simple_getchu")
tf.app.flags.DEFINE_boolean("side_noise", False, "If to add side noise")
tf.app.flags.DEFINE_integer("gpu", 6, "which gpu to use")
FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    # get configuration
    print("Get configuration")
    TFLAGS = cfg.get_train_config(FLAGS.model_name)

    gen_config = cfg.good_generator.goodmodel_gen({})
    gen_config['name'] = "CondDeepGenerator"
    gen_model = model.conditional_generator.ImageConditionalDeepGenerator2(**gen_config)
    disc_config = cfg.good_generator.goodmodel_disc({})
    disc_config['norm_mtd'] = None
    disc_model = model.good_generator.GoodDiscriminator(**disc_config)
    
    TFLAGS['batch_size'] = 1
    TFLAGS['AE_weight'] = 10.0
    TFLAGS['side_noise'] = FLAGS.side_noise
    TFLAGS['input_shape'] = [128, 128, 3]
    # Data Preparation
    # raw image is 0~255
    print("Get dataset")

    face_dataset = dataloader.CustomDataset(
        root_dir="/data/datasets/getchu/true_face",
        npy_dir=None,
        preproc_kind="tanh",
        img_size=TFLAGS['input_shape'][:2],
        filter_data=TFLAGS['filter_data'],
        class_num=TFLAGS['c_len'],
        disturb=False,
        flip=False)

    sketch_dataset = dataloader.CustomDataset(
        root_dir="/data/datasets/getchu/sketch_face/",
        npy_dir=None,
        preproc_kind="tanh",
        img_size=TFLAGS['input_shape'][:2],
        filter_data=TFLAGS['filter_data'],
        class_num=TFLAGS['c_len'],
        disturb=False,
        flip=False)
    
    label_dataset = dataloader.NumpyArrayDataset(np.load("/data/datasets/getchu/true_face.npy")[:, 1:])

    listsampler = dataloader.ListSampler([sketch_dataset, face_dataset, label_dataset])
    dl = dataloader.CustomDataLoader(listsampler, TFLAGS['batch_size'], 4)

    # TF Input
    x_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    s_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'][:-1] + [1], name='s_real')
    fake_x_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    noise_A = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="noise_A")
    noise_B = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="noise_B")
    label_A = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="label_A")
    label_B = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="label_B")

    # c label is for real samples
    c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")
    c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")

    # control variables
    real_cls_weight = tf.placeholder(tf.float32, [], name="real_cls_weight")
    fake_cls_weight = tf.placeholder(tf.float32, [], name="fake_cls_weight")
    adv_weight = tf.placeholder(tf.float32, [], name="adv_weight")
    lr = tf.placeholder(tf.float32, [], name="lr")

    side_noise_A = tf.concat([noise_A, c_label], axis=1, name='side_noise_A')

    if TFLAGS['side_noise']:
        gen_model.side_noise = side_noise_A
    x_fake = gen_model([s_real]); gen_model.set_reuse()
    gen_model.x_fake = x_fake

    if FLAGS.cgan:
        disc_model.is_cgan = True
    disc_model.disc_real_out, disc_model.real_cls_logits = disc_model(x_real)[:2]
    disc_model.set_reuse()
    disc_model.disc_fake_out, disc_model.fake_cls_logits = disc_model(x_fake)[:2]

    gen_model.cost = disc_model.cost = 0
    gen_model.sum_op = disc_model.sum_op = []
    inter_sum_op = []

    if TFLAGS['gan_loss'] == "dra":
        # naive GAN loss
        gen_cost_, disc_cost_, gen_sum_, disc_sum_ = loss.naive_ganloss.get_naive_ganloss(
            gen_model, disc_model, adv_weight)
        gen_model.cost += gen_cost_; disc_model.cost += disc_cost_
        gen_model.sum_op.extend(gen_sum_)
        disc_model.sum_op.extend(disc_sum_)

        gen_model.gen_cost = gen_cost_
        disc_model.disc_cost = disc_cost_

        # dragan loss
        disc_cost_, disc_sum_ = loss.dragan_loss.get_dragan_loss(disc_model, x_real, TFLAGS['gp_weight'])
        disc_model.cost += disc_cost_
        disc_model.sum_op.append(disc_sum_)

    elif TFLAGS['gan_loss'] == "wass":
        gen_cost_, disc_cost_, gen_sum_, disc_sum_ = loss.wass_ganloss.wass_gan_loss(gen_model, disc_model, x_real, x_fake)
        gen_model.cost += gen_cost_; disc_model.cost += disc_cost_
        gen_model.sum_op.extend(gen_sum_)
        disc_model.sum_op.extend(disc_sum_)

    elif TFLAGS['gan_loss'] == "naive":
        # naive GAN loss
        gen_cost_, disc_cost_, gen_sum_, disc_sum_ = loss.naive_ganloss.get_naive_ganloss(
            gen_model, disc_model, adv_weight, lsgan=True)
        gen_model.cost += gen_cost_; disc_model.cost += disc_cost_
        gen_model.sum_op.extend(gen_sum_)
        disc_model.sum_op.extend(disc_sum_)
        
        gen_model.gen_cost = gen_cost_
        disc_model.disc_cost = disc_cost_
    
    if FLAGS.cgan:
        real_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_model.real_cls_logits,
            labels=c_label), name="real_cls_reduce_mean") * real_cls_weight

        #fake_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=disc_model.fake_cls_logits,
        #    labels=c_label), name="fake_cls_reduce_mean") * fake_cls_weight

        disc_model.cost += real_cls# + fake_cls
        disc_model.sum_op.extend([
            tf.summary.scalar("RealCls", real_cls)])#,
            #tf.summary.scalar("FakeCls", fake_cls)])

    gen_cost_, ae_loss_sum_ = loss.common_loss.reconstruction_loss(x_fake, x_real, TFLAGS['AE_weight'])
    gen_model.sum_op.append(ae_loss_sum_)
    gen_model.cost += gen_cost_

    gen_model.cost = tf.identity(gen_model.cost, "TotalGenCost")
    disc_model.cost = tf.identity(disc_model.cost, "TotalDiscCost")
    
    # total summary
    gen_model.sum_op.append(tf.summary.scalar("GenCost", gen_model.cost))
    disc_model.sum_op.append(tf.summary.scalar("DiscCost", disc_model.cost))

    # add interval summary
    edge_num = int(np.sqrt(TFLAGS['batch_size']))
    if edge_num > 4:
        edge_num = 4
    grid_x_fake = ops.get_grid_image_summary(x_fake, edge_num)
    inter_sum_op.append(tf.summary.image("generated image", grid_x_fake))
    grid_x_real = ops.get_grid_image_summary(x_real, edge_num)
    grid_x_real = tf.Print(grid_x_real, [tf.reduce_max(grid_x_real), tf.reduce_min(grid_x_real)], "Real")
    inter_sum_op.append(tf.summary.image("real image", grid_x_real))
    inter_sum_op.append(tf.summary.image("sketch image", ops.get_grid_image_summary(s_real, edge_num) ))

    # merge summary op
    gen_model.sum_op = tf.summary.merge(gen_model.sum_op)
    disc_model.sum_op = tf.summary.merge(disc_model.sum_op)
    inter_sum_op = tf.summary.merge(inter_sum_op)

    # get train op
    gen_model.get_trainable_variables()
    disc_model.get_trainable_variables()
    
    # Not applying update op will result in failure
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gen_model.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.5,
            beta2=0.9).minimize(gen_model.cost, var_list=gen_model.vars)
        
        disc_model.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.5,
            beta2=0.9).minimize(disc_model.cost, var_list=disc_model.vars)

    print("=> ##### Generator Variable #####")
    gen_model.print_trainble_vairables()
    print("=> ##### Discriminator Variable #####")
    disc_model.print_trainble_vairables()
    print("=> ##### All Variable #####")
    for v in tf.trainable_variables():
        print("%s" % v.name)

    ctrl_weight = {
            "real_cls"  : real_cls_weight,
            "fake_cls"  : fake_cls_weight,
            "adv"       : adv_weight,
            "lr"        : lr
        }

    trainer_feed = {
        "gen_model": gen_model,
        "disc_model" : disc_model,
        "noise": noise_A,
        "ctrl_weight": ctrl_weight,
        "dataloader": dl,
        "int_sum_op": inter_sum_op,
        "gpu_mem": TFLAGS['gpu_mem'],
        "FLAGS": FLAGS,
        "TFLAGS": TFLAGS
    }

    Trainer = trainer.cgantrainer.CGANTrainer
    trainer_feed.update({
        "inputs": [s_real, x_real, c_label],
    })
    
    ModelTrainer = Trainer(**trainer_feed)

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()

    disc_model.load_from_npz('success/goodmodel_dragan_anime1_disc.npz', ModelTrainer.sess)

    ModelTrainer.train()

if __name__ == "__main__":
    main()
