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

BEST_MODEL = "success/goodmodel_fulldata_dragan/goodmodel_dragan_anime1/goodmodel_dragan_anime1"

tf.app.flags.DEFINE_string("rpath", BEST_MODEL, "The path to npz param")
tf.app.flags.DEFINE_boolean("debug", False, "Whether to debug")
tf.app.flags.DEFINE_boolean("reload", False, "If to reload from npz param. Incompatible with --resume.")
tf.app.flags.DEFINE_string("resume", False, "If to resume from previous expr. Incompatible with --reload.")
tf.app.flags.DEFINE_boolean("cgan", False, "If use ACGAN")
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

    gen_config = cfg.good_generator.deepmodel_gen({})
    disc_config = cfg.good_generator.deepmodel_disc({})
    gen_model = model.good_generator.DeepGenerator(**gen_config)
    disc_model = model.good_generator.DeepDiscriminator(**disc_config)

    # Data Preparation
    # raw image is 0~255
    print("Get dataset")
    if TFLAGS['dataset_kind'] == "file":
        dataset = dataloader.CustomDataset(
            root_dir=TFLAGS['data_dir'],
            npy_dir=TFLAGS['npy_dir'],
            preproc_kind=TFLAGS['preprocess'],
            img_size=TFLAGS['input_shape'][:2],
            filter_data=TFLAGS['filter_data'],
            class_num=TFLAGS['c_len'],
            disturb=TFLAGS['disturb'],
            flip=TFLAGS['preproc_flip'])

    elif TFLAGS['dataset_kind'] == "numpy":
        train_data, test_data, train_label, test_label = utils.load_mnist_4d(TFLAGS['data_dir'])
        train_data = train_data.reshape([-1] + TFLAGS['input_shape'])

        # TODO: validation in training
        dataset = dataloader.NumpyArrayDataset(
            data_npy=train_data,
            label_npy=train_label,
            preproc_kind=TFLAGS['preprocess'],
            img_size=TFLAGS['input_shape'][:2],
            filter_data=TFLAGS['filter_data'],
            class_num=TFLAGS['c_len'],
            flip=TFLAGS['preproc_flip'])
    
    elif TFLAGS['dataset_kind'] == "fuel":
        dataset = dataloader.FuelDataset(
            hdfname=TFLAGS['data_dir'],
            npy_dir=TFLAGS['npy_dir'],
            preproc_kind=TFLAGS['preprocess'],
            img_size=TFLAGS['input_shape'][:2],
            filter_data=TFLAGS['filter_data'],
            class_num=TFLAGS['c_len'],
            flip=TFLAGS['preproc_flip'])
    
    dl = dataloader.CustomDataLoader(dataset, batch_size=TFLAGS['batch_size'], num_threads=TFLAGS['data_threads'])

    # TF Input
    x_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    s_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name='s_real')
    z_noise = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="z_noise")
    
    if FLAGS.cgan:
        c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
        c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # Select loss builder and model trainer
    print("Build training graph")
    if TFLAGS['gan_loss'].find("naive") > -1:
        loss_builder_ = loss.naive_ganloss.NaiveGANLoss
        Trainer = trainer.base_gantrainer.BaseGANTrainer
    elif TFLAGS['gan_loss'].find("wass") > -1:
        loss_builder_ = loss.wass_ganloss.WGANLoss
        Trainer = trainer.base_gantrainer.BaseGANTrainer
    elif TFLAGS['gan_loss'].find("dra") > -1:
        loss_builder_ = loss.dragan_loss.DRAGANLoss
        Trainer = trainer.base_gantrainer.BaseGANTrainer
    elif TFLAGS['gan_loss'].find("ian") > -1:
        loss_builder_ = loss.ian_loss.IANLoss
        Trainer = trainer.ian_trainer.IANTrainer

    if FLAGS.cgan:
        loss_builder = loss_builder_(
            gen_model=gen_model,
            disc_model=disc_model,
            gen_inputs=[z_noise, c_noise],
            real_inputs=[x_real, c_label],
            has_ac=True,
            **TFLAGS)
    else:
        loss_builder = loss_builder_(
            gen_model=gen_model,
            disc_model=disc_model,
            gen_inputs=[z_noise],
            real_inputs=[x_real],
            has_ac=False,
            **TFLAGS)

    loss_builder.build()
    int_sum_op = tf.summary.merge(loss_builder.inter_sum_op)
    loss_builder.get_trainable_variables()
    
    print("=> ##### Generator Variable #####")
    gen_model.print_trainble_vairables()
    print("=> ##### Discriminator Variable #####")
    disc_model.print_trainble_vairables()
    print("=> ##### All Variable #####")
    for v in tf.trainable_variables():
        print("%s" % v.name)

    if FLAGS.cgan:
        inputs = [z_noise, c_noise, x_real, c_label]
        if TFLAGS['gan_loss'].find("ian") > -1:
            ctrl_weight={
                "real_cls"  : loss_builder.real_cls_weight,
                "fake_cls"  : loss_builder.fake_cls_weight,
                "rec_weight": loss_builder.rec_weight,
                "adv"       : loss_builder.adv_weight,
                "recadv_weight"    : loss_builder.recadv_weight
            }
        else:
            ctrl_weight = {
                "real_cls"  : loss_builder.real_cls_weight,
                "fake_cls"  : loss_builder.fake_cls_weight,
                "adv"       : loss_builder.adv_weight
            }
    else:
        inputs = [z_noise, x_real]
        ctrl_weight = {
                "real_cls"  : loss_builder.real_cls_weight,
                "fake_cls"  : loss_builder.fake_cls_weight,
                "adv"       : loss_builder.adv_weight
            }

    ModelTrainer = Trainer(
        gen_model=gen_model,
        disc_model=disc_model,
        inputs=inputs,
        dataloader=dl,
        int_sum_op=int_sum_op,
        ctrl_weight=ctrl_weight,
        gpu_mem=TFLAGS['gpu_mem'],
        FLAGS=FLAGS,
        TFLAGS=TFLAGS
    )

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()
    ModelTrainer.train()

if __name__ == "__main__":
    main()
