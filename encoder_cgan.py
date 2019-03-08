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
tf.app.flags.DEFINE_boolean("cgan", True, "Must be true")
tf.app.flags.DEFINE_boolean("cache", False, "If to use cache")
tf.app.flags.DEFINE_string("model_name", "simple_getchu", "model type: simple_getchu")
tf.app.flags.DEFINE_integer("gpu", 6, "which gpu to use")
FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    # get configuration
    print("Get configuration")
    TFLAGS = cfg.get_train_config(FLAGS.model_name)
    gen_model, gen_config, disc_model, disc_config = model.get_model(FLAGS.model_name, TFLAGS)
    gen_model = model.conditional_generator.ImageConditionalEncoder()
    disc_model = model.conditional_generator.ImageConditionalDeepDiscriminator()
    gen_model.name = "ImageConditionalEncoder"
    disc_model.name = "ImageConditionalDeepDiscriminator"
    print("Common length: %d" % gen_model.common_length)
    TFLAGS['AE_weight'] = 1.0
    TFLAGS['batch_size'] = 1
    TFLAGS['input_shape'] = [256, 256, 3]

    face_dataset = dataloader.CustomDataset(
        root_dir="/data/datasets/getchu/crop_character",
        npy_dir="/data/datasets/getchu/true_character.npy",
        preproc_kind="tanh",
        img_size=TFLAGS['input_shape'][:2],
        filter_data=TFLAGS['filter_data'],
        class_num=TFLAGS['c_len'],
        has_gray=False,
        disturb=False,
        flip=False)

    sketch_dataset = dataloader.CustomDataset(
        root_dir="/data/datasets/getchu/crop_sketch_character/",
        npy_dir=None,
        preproc_kind="tanh",
        img_size=TFLAGS['input_shape'][:2],
        filter_data=TFLAGS['filter_data'],
        class_num=TFLAGS['c_len'],
        has_gray=True,
        disturb=False,
        flip=False)
    
    label_dataset = dataloader.NumpyArrayDataset(np.load("/data/datasets/getchu/true_face.npy")[:, 1:])

    listsampler = dataloader.ListSampler([face_dataset, sketch_dataset, label_dataset])
    dl = dataloader.CustomDataLoader(listsampler, TFLAGS['batch_size'], 4)

    # TF Input
    x_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    fake_x_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    # semantic segmentation
    s_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'][:-1] + [1,], name='s_real')
    z_noise = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="z_noise")
    c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
    c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # control variables
    real_cls_weight = tf.placeholder(tf.float32, [], name="real_cls_weight")
    fake_cls_weight = tf.placeholder(tf.float32, [], name="fake_cls_weight")
    adv_weight = tf.placeholder(tf.float32, [], name="adv_weight")
    inc_length = tf.placeholder(tf.int32, [], name="inc_length")
    lr = tf.placeholder(tf.float32, [], name="lr")

    with tf.variable_scope(gen_model.name):
        gen_model.image_input = x_real
        gen_model.seg_input = s_real
        gen_model.noise_input = tf.concat([z_noise, c_noise], axis=1)

        seg_image, image_image, seg_seg, image_seg = gen_model.build_inference()
        seg_feat = tf.identity(gen_model.seg_feat, "seg_feat")
        image_feat = tf.identity(gen_model.image_feat, "image_feat")

        gen_model.x_fake = tf.identity(seg_image)

    disc_real_out = disc_model([x_real, s_real]); disc_model.set_reuse()
    disc_fake_out = disc_model([seg_image, s_real])
    disc_fake_sketch_out = disc_model([x_real, image_seg])
    #disc_rec_out  = disc_model([image_image, s_real])
    disc_fake_sample_out  = disc_model([fake_x_sample, s_real])

    gen_model.cost = disc_model.cost = 0
    gen_model.sum_op = disc_model.sum_op = []
    inter_sum_op = []

    # Select loss builder and model trainer
    print("Build training graph")

    # Naive GAN
    loss.naive_ganloss.func_gen_loss(disc_fake_out, adv_weight * 0.9, name="GenSeg", model=gen_model)
    loss.naive_ganloss.func_gen_loss(disc_fake_sketch_out, adv_weight * 0.1, name="GenSketch", model=gen_model)
    if FLAGS.cache:
        loss.naive_ganloss.func_disc_fake_loss(disc_fake_sample_out, adv_weight * 0.9, name="DiscFake", model=disc_model)
        loss.naive_ganloss.func_disc_fake_loss(disc_fake_sample_out, adv_weight * 0.1, name="DiscFakeSketch", model=disc_model)
    else:
        loss.naive_ganloss.func_disc_fake_loss(disc_fake_out, adv_weight * 0.9, name="DiscFake", model=disc_model)
        loss.naive_ganloss.func_disc_fake_loss(disc_fake_sketch_out, adv_weight * 0.1, name="DiscFakeSketch", model=disc_model)
        
    loss.naive_ganloss.func_disc_real_loss(disc_real_out, adv_weight, name="DiscGen", model=disc_model)

    # recontrust of sketch
    rec_sketch_cost_, rec_sketch_sum_ = loss.common_loss.reconstruction_loss(
        image_seg, s_real, TFLAGS['AE_weight'], name="RecSketch")
    gen_model.cost += rec_sketch_cost_
    gen_model.sum_op.append(rec_sketch_sum_)

    gen_model.cost = tf.identity(gen_model.cost, "TotalGenCost")
    disc_model.cost = tf.identity(disc_model.cost, "TotalDiscCost")
    
    # total summary
    gen_model.sum_op.append(tf.summary.scalar("GenCost", gen_model.cost))
    disc_model.sum_op.append(tf.summary.scalar("DiscCost", disc_model.cost))

    # add interval summary
    edge_num = int(np.sqrt(TFLAGS['batch_size']))
    if edge_num > 4:
        edge_num = 4
    grid_x_fake = ops.get_grid_image_summary(seg_image, edge_num)
    inter_sum_op.append(tf.summary.image("generated image", grid_x_fake))
    grid_x_seg = ops.get_grid_image_summary(image_seg, edge_num)
    inter_sum_op.append(tf.summary.image("inv image", grid_x_seg))
    grid_x_real = ops.get_grid_image_summary(x_real, edge_num)
    inter_sum_op.append(tf.summary.image("real image", grid_x_real))
    grid_s_real = ops.get_grid_image_summary(s_real, edge_num)
    inter_sum_op.append(tf.summary.image("sketch image", grid_s_real))

    # merge summary op
    gen_model.sum_op = tf.summary.merge(gen_model.sum_op)
    disc_model.sum_op = tf.summary.merge(disc_model.sum_op)
    inter_sum_op = tf.summary.merge(inter_sum_op)

    print("=> Compute gradient")

    # get train op
    gen_model.get_trainable_variables()
    disc_model.get_trainable_variables()
    
    # Not applying update op will result in failure
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # normal GAN train op
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
            "lr"        : lr,
            "inc_length": inc_length
        }

    trainer_feed = {
        "gen_model": gen_model,
        "disc_model" : disc_model,
        "ctrl_weight": ctrl_weight,
        "dataloader": dl,
        "int_sum_op": inter_sum_op,
        "gpu_mem": TFLAGS['gpu_mem'],
        "FLAGS": FLAGS,
        "TFLAGS": TFLAGS
    }

    Trainer = trainer.cgantrainer.CGANTrainer
    Trainer.use_cache = False
    trainer_feed.update({
        "inputs": [x_real, s_real, c_label],
        "noise": z_noise,
        "cond": c_noise
    })

    ModelTrainer = Trainer(**trainer_feed)
    ModelTrainer.fake_x_sample = fake_x_sample

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()

    ModelTrainer.train()

if __name__ == "__main__":
    main()
