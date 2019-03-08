"""
Two stage GAN training pipeline. (Currently this is second stage)
GAN takes in real data and sketch & mask pair, try to modify it!
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

BEST_MODEL = "success/goodmodel_dragan_anime1/goodmodel_dragan_anime1"

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to debug")
tf.app.flags.DEFINE_boolean("cgan", False, "If to use condition variable")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_string("resume", False, "If to resume from previous expr. Incompatible with --reload.")
tf.app.flags.DEFINE_string("model_name", "condsim_dragan_anime", "model type: simple_getchu")
tf.app.flags.DEFINE_integer("mode", 1, "1 - naive; 2 - sketch; 3 - ae")
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
    TFLAGS['AE_weight'] = 100.0
    TFLAGS['side_noise'] = False
    TFLAGS['input_shape'] = [128, 128, 3]

    """
    TFLAGS = cfg.get_train_config(FLAGS.model_name)
    gen_config = {}
    disc_config = {}
    cfg.simple_generator.simple_gen(gen_config)
    cfg.simple_generator.simple_disc(disc_config)
    print("=> Generator config")
    pprint.pprint(gen_config)
    print("=> Discriminator config")
    pprint.pprint(disc_config)

    if FLAGS.model_name.find("goodmodel") > -1:
        gen_model = model.conditional_generator.ImageConditionalDeepGenerator(**gen_config)
        disc_model = model.good_generator.GoodDiscriminator(**disc_config)
        #disc_model = model.conditional_generator.ImageConditionalDeepDiscriminator(**disc_config)
    else:
        gen_model = model.conditional_generator.ImageConditionalGenerator(**gen_config)
        disc_model = model.conditional_generator.ImageConditionalDiscriminator(**disc_config)
    """

    # Data Preparation
    # raw image is 0~255
    print("=> Get dataset")
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

    mask_dataset = line_dataset = 0
    if FLAGS.mode == 2:
        print("=> Read mask & sketch into memory")
        # [HardCoded] Dir for sketch & mask dir
        line_mask_data = np.array(dataloader.ReadAllDataset("/data/datasets/nim/sketchdata", (128, 128)).get_data())
        mask_arr = line_mask_data[::2, :, :, :]
        line_arr = line_mask_data[1::2, :, :, :] * 2.0 - 1
        mask_dataset = dataloader.NumpyArrayDataset(mask_arr)
        line_dataset = dataloader.NumpyArrayDataset(line_arr)
        #sketch_mask_sampler = dataloader.MaskSampler(sketch_mask_data)

    dl = dataloader.CustomDataLoader(dataloader.ListSampler([sketch_dataset, face_dataset, label_dataset, mask_dataset, line_dataset]), TFLAGS['batch_size'], 4)

    # TF Input
    x_real = tf.placeholder(tf.float32, [None, 128, 128, 1], name="x_real")
    f_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="f_real")
    s_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="s_real")
    m_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="m_real")
    z_noise = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="z_noise")
    c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
    c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # control variables
    real_cls_weight = tf.placeholder(tf.float32, [], name="real_cls_weight")
    fake_cls_weight = tf.placeholder(tf.float32, [], name="fake_cls_weight")
    adv_weight = tf.placeholder(tf.float32, [], name="adv_weight")
    lr = tf.placeholder(tf.float32, [], name="lr")

    # build model
    if FLAGS.mode == 3:
        # AE
        gen_model.gen_out = gen_model([x_real])
    else:
        gen_model.gen_out = gen_model([m_real, s_real, x_real])

    x_fake = gen_model.gen_out
    gen_model.set_reuse()

    disc_model.disc_real_out, disc_model.real_cls_logits = disc_model(f_real)[:2]
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
            gen_model, disc_model, adv_weight, lsgan=False)
        gen_model.cost += gen_cost_; disc_model.cost += disc_cost_
        gen_model.sum_op.extend(gen_sum_)
        disc_model.sum_op.extend(disc_sum_)
        
        gen_model.gen_cost = gen_cost_
        disc_model.disc_cost = disc_cost_


    # sketch loss
    if FLAGS.mode == 2:
        mask_area = tf.reduce_mean(m_real, [1, 2, 3], keep_dims=True) + 1e-6

        gen_cost_, sketch_loss_sum_ = loss.common_loss.mask_reconstruction_loss(x_fake, m_real, s_real, 1/mask_area * TFLAGS['AE_weight'])
        gen_model.sum_op.append(sketch_loss_sum_)
        gen_model.cost += gen_cost_

        gen_cost_, ae_loss_sum_ = loss.common_loss.mask_reconstruction_loss(x_fake, 1-m_real, f_real, 1/(1-mask_area) * TFLAGS['AE_weight'])
        gen_model.sum_op.append(ae_loss_sum_)
        gen_model.cost += gen_cost_
    
    # AE mode
    if FLAGS.mode == 3:
        gen_cost_, ae_loss_sum_ = loss.common_loss.reconstruction_loss(x_fake, x_real, 100.0)
        gen_model.sum_op.append(ae_loss_sum_)
        gen_model.cost += gen_cost_
    
    if FLAGS.cgan:
        real_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_model.real_cls_logits,
            labels=c_label), name="real_cls_reduce_mean")
        
        """
        fake_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_model.fake_cls_logits,
            labels=c_noise), name='fake_cls_reduce_mean')
        """

        disc_model.cost += real_cls #+ fake_cls
        #gen_model.cost += fake_cls
        disc_model.sum_op.extend([
            tf.summary.scalar("RealCls", real_cls)])
            #tf.summary.scalar("FakeCls", fake_cls)])

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
    grid_x_real = ops.get_grid_image_summary(s_real * m_real + x_real * (1-m_real), edge_num)#
    inter_sum_op.append(tf.summary.image("real image", grid_x_real))
    inter_sum_op.append(tf.summary.image("mask image", ops.get_grid_image_summary(m_real, edge_num) ))
    inter_sum_op.append(tf.summary.image("sketch image", ops.get_grid_image_summary(s_real, edge_num) ))

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
        
    # merge summary op
    gen_model.sum_op = tf.summary.merge(gen_model.sum_op)
    disc_model.sum_op = tf.summary.merge(disc_model.sum_op)
    inter_sum_op = tf.summary.merge(inter_sum_op)

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
        "ctrl_weight": ctrl_weight,
        "dataloader": dl,
        "int_sum_op": inter_sum_op,
        "gpu_mem": TFLAGS['gpu_mem'],
        "FLAGS": FLAGS,
        "TFLAGS": TFLAGS
    }

    Trainer = trainer.cgantrainer.CGANTrainer
    if FLAGS.mode == 2:
        trainer_feed.update({
            "inputs": [x_real, f_real, c_label, m_real, s_real],
        })
    elif FLAGS.mode == 3:
        Trainer = trainer.autoencoder_trainer.AETrainer
        trainer_feed.update({
            "inputs": [x_real, c_label]
        })
    
    ModelTrainer = Trainer(**trainer_feed)

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()

    disc_model.load_from_npz('success/goodmodel_dragan_anime1_disc.npz', ModelTrainer.sess)


    ModelTrainer.train()

if __name__ == "__main__":
    main()
