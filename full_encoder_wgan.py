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
tf.app.flags.DEFINE_string("model_name", "simple_getchu", "model type: simple_getchu")
tf.app.flags.DEFINE_integer("gpu", 6, "which gpu to use")
FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def main():
    # get configuration
    print("=> Get configuration")
    TFLAGS = cfg.get_train_config(FLAGS.model_name)
    gen_model, gen_config, disc_model, disc_config = model.get_model(FLAGS.model_name, TFLAGS)
    gen_model = model.good_generator.GoodGeneratorEncoder(**gen_config)
    print("=> Common length: %d" % gen_model.common_length)
    TFLAGS['AE_weight'] = 1.0
    TFLAGS['nfeat_weight'] = 1.0
    TFLAGS['tfeat_weight'] = 0.25

    # Data Preparation
    # raw image is 0~255
    print("=> Get dataset")
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
    
    dl = dataloader.CustomDataLoader(dataset, batch_size=TFLAGS['batch_size'], num_threads=TFLAGS['data_threads'])

    # TF Input
    x_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="x_real")
    s_real = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name='s_real')
    z_noise = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="z_noise")
    
    if FLAGS.cgan:
        c_noise = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_noise")
        c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # control variables
    real_cls_weight = tf.placeholder(tf.float32, [], name="real_cls_weight")
    fake_cls_weight = tf.placeholder(tf.float32, [], name="fake_cls_weight")
    adv_weight = tf.placeholder(tf.float32, [], name="adv_weight")
    inc_length = tf.placeholder(tf.int32, [], name="inc_length")
    lr = tf.placeholder(tf.float32, [], name="lr")

    # build inference network

    # image_inputs : x_real, x_fake, x_trans
    # noise_input : noise, noise_rec, noise_trans

    # feat_image + image_feat
    # x_real -> trans_feat -> x_trans -> trans_x_trans_feat[REC] -> x_trans_trans_

    # feat_noise + noise_feat
    # x_real -> trans_feat -> noise_trans -> noise_x_trans_feat[REC] -> x_trans_fake
    print("=> Inference Net")
    with tf.variable_scope(gen_model.field_name):
        noise_input = tf.concat([z_noise, c_noise], axis=1)
        gen_model.image_input = x_real
        gen_model.noise_input = noise_input
        
        x_fake, x_trans, noise_rec, noise_trans = gen_model.build_inference()
        noise_feat = tf.identity(gen_model.noise_feat, "noise_feat")
        trans_feat = tf.identity(gen_model.image_feat, "trans_feat")
        
        gen_model.noise_input = noise_rec
        gen_model.image_input = x_fake

        x_fake_fake, x_fake_trans, noise_rec_rec, noise_x_fake_trans = gen_model.build_inference()
        noise_rec_feat = tf.identity(gen_model.noise_feat, "noise_rec_feat")
        trans_fake_feat = tf.identity(gen_model.image_feat, "trans_fake_feat")
        
        gen_model.noise_input = noise_trans
        gen_model.image_input = x_trans

        x_trans_fake, x_trans_trans, noise_trans_rec, noise_trans_trans = gen_model.build_inference()
        noise_trans_feat = tf.identity(gen_model.noise_feat, "noise_trans_feat")
        trans_trans_feat = tf.identity(gen_model.image_feat, "trans_trans_feat")

    # noise -> noise_feat -> x_fake | noise_rec
    # image -> trans_feat -> x_trans | noise_trans
    # noise_rec -> noise_rec_feat -> x_fake_fake | noise_rec_rec
    # x_fake -> trans_fake_feat -> x_fake_trans | noise_fake_trans
    # noise_trans -> noise_trans_feat -> x_trans_fake | noise_trans_rec    
    # x_trans -> trans_trans_feat -> x_trans_trans | noise_trans_trans

    # Image Feat Recs:
    # trans_feat == noise_trans_feat : feat_noise + noise_feat
    # trans_feat == trans_trans_feat : feat_image + image_feat
    # noise_feat == noise_rec_feat : feat_noise + noise_feat
    # noise_feat == trans_fake_feat : feat_image + image_feat

    # Noise Recs:
    # noise_input == noise_rec : noise_feat + feat_noise
    
    # Image Recs:
    # x_real -[REC]- x_trans : image_feat + feat_image

    disc_model.disc_real_out, cls_real_logits = disc_model(x_real)[:2]; disc_model.set_reuse()
    disc_model.disc_fake_out, cls_fake_logits = disc_model(x_fake)[:2]
    disc_model.disc_trans_out, cls_trans_logits = disc_model(x_trans)[:2]

    gen_model.cost = disc_model.cost = 0
    gen_model.sum_op = disc_model.sum_op = []
    inter_sum_op = []

    # Select loss builder and model trainer
    print("=> Build train graph")

    # Wass GAN
    loss.wass_ganloss.func_wass_loss(gen_model, disc_model, x_real, x_fake, adv_weight / 2, name="NoiseGen")
    gen_cost_single = tf.identity(gen_model.cost)
    loss.wass_ganloss.func_wass_loss(gen_model, disc_model, x_real, x_trans, adv_weight / 2, name="TransGen")

    # ACGAN
    real_cls_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=cls_real_logits,
        labels=c_label), name="real_cls_reduce_mean")
    real_cls_cost_sum = tf.summary.scalar("real_cls", real_cls_cost)
    
    fake_cls_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=cls_fake_logits,
        labels=c_noise), name='fake_cls_reduce_mean')
    fake_cls_cost_sum = tf.summary.scalar("fake_cls", fake_cls_cost)
    
    disc_model.cost += real_cls_cost * real_cls_weight + fake_cls_cost * fake_cls_weight
    disc_model.sum_op.extend([real_cls_cost_sum, fake_cls_cost_sum])

    gen_model.cost += fake_cls_cost * fake_cls_weight

    # DRAGAN gradient penalty
    disc_cost_, disc_sum_ = loss.dragan_loss.get_dragan_loss(disc_model, x_real, TFLAGS['gp_weight'])
    disc_model.cost += disc_cost_
    disc_model.sum_op.append(disc_sum_)

    # Image Feat Recs:
    # trans_feat == noise_trans_feat : feat_noise + noise_feat 1
    # trans_feat == trans_trans_feat : feat_image + image_feat 2
    # noise_feat == noise_rec_feat : feat_noise + noise_feat 1
    # noise_feat == trans_fake_feat : feat_image + image_feat 2
    trans_feat_cost1_, _ = loss.common_loss.reconstruction_loss(
        noise_trans_feat, trans_feat, TFLAGS['nfeat_weight'], name="RecTransFeat1")
    trans_feat_cost2_, _ = loss.common_loss.reconstruction_loss(
        trans_trans_feat, trans_feat, TFLAGS['tfeat_weight'], name="RecTransFeat2")
    trans_feat_cost_ = trans_feat_cost1_ + trans_feat_cost2_
    trans_feat_sum_ = tf.summary.scalar("RecTransFeat", trans_feat_cost_)

    noise_feat_cost1_, _ = loss.common_loss.reconstruction_loss(
        noise_rec_feat, noise_feat, TFLAGS['nfeat_weight'], name="RecNoiseFeat1")
    noise_feat_cost2_, _ = loss.common_loss.reconstruction_loss(
        trans_fake_feat, noise_feat, TFLAGS['tfeat_weight'], name="RecNoiseFeat2")
    noise_feat_cost_ = noise_feat_cost1_ + noise_feat_cost2_
    noise_feat_sum_ = tf.summary.scalar("RecNoiseFeat", noise_feat_cost_)

    # Noise Recs:
    # noise_input -[REC]- noise_rec : noise_feat + feat_noise
    noise_cost_, noise_sum_ = loss.common_loss.reconstruction_loss(
        noise_rec[:, -inc_length:], noise_input[:, -inc_length:], TFLAGS['AE_weight'], name="RecNoise")

    # Image Recs:
    # x_real -[REC]- x_trans : image_feat + feat_image
    rec_cost_, rec_sum_ = loss.common_loss.reconstruction_loss(
        x_trans, x_real, TFLAGS['AE_weight'], name="RecPix")

    gen_model.sum_op.extend([trans_feat_sum_, noise_feat_sum_, rec_sum_, noise_sum_])
    gen_model.cost += trans_feat_cost_ + noise_feat_cost_ + rec_cost_ + noise_cost_

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
    grid_x_trans = ops.get_grid_image_summary(x_trans, edge_num)
    inter_sum_op.append(tf.summary.image("trans image", grid_x_trans))
    grid_x_real = ops.get_grid_image_summary(x_real, edge_num)
    inter_sum_op.append(tf.summary.image("real image", grid_x_real))

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
        # train noise 
        noise_branch_vars = gen_model.branch_vars['noise_feat'] + gen_model.branch_vars['feat_noise']
        gen_model.extra_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.5,
            beta2=0.9).minimize(gen_cost_single, var_list=noise_branch_vars)

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

    Trainer = trainer.base_gantrainer_full.BaseGANTrainer
    trainer_feed.update({
        "inputs": [z_noise, c_noise, x_real, c_label],
    })
    
    ModelTrainer = Trainer(**trainer_feed)

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()

    ModelTrainer.train()

if __name__ == "__main__":
    main()
