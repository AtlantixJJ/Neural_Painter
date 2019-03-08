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
from lib import utils, dataloader, ops, files

BEST_MODEL = "success/goodmodel_dragan_anime1/goodmodel_dragan_anime1"

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
    TFLAGS['AE_weight'] = 10.0
    TFLAGS['side_noise'] = False

    # A: sketch face, B: anime face
    # Gen and Disc usually init from pretrained model
    print("=> Building discriminator")
    disc_config = cfg.good_generator.goodmodel_disc({})
    disc_config['name'] = "DiscA"
    DiscA = model.good_generator.GoodDiscriminator(**disc_config)
    disc_config['name'] = "DiscB"
    DiscB = model.good_generator.GoodDiscriminator(**disc_config)

    gen_config = cfg.good_generator.goodmodel_gen({})
    gen_config['name'] = 'GenA'
    gen_config['out_dim'] = 3
    GenA = model.good_generator.GoodGenerator(**gen_config)
    gen_config['name'] = 'GenB'
    gen_config['out_dim'] = 1
    GenB = model.good_generator.GoodGenerator(**gen_config)

    gen_config = {}
    gen_config['name'] = 'TransForward'
    gen_config['out_dim'] = 3
    TransF = model.conditional_generator.ImageConditionalDeepGenerator2(**gen_config)
    gen_config['name'] = 'TransBackward'
    gen_config['out_dim'] = 1
    TransB = model.conditional_generator.ImageConditionalDeepGenerator2(**gen_config)

    models = [DiscA, DiscB, TransF, TransB]
    model_names = ["DiscA", "DiscB", "TransF", "TransB"]

    TFLAGS['batch_size'] = 1
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
    real_A_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'][:-1] + [1], name='real_A_sample')
    real_B_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="real_B_sample")
    fake_A_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'][:-1] + [1], name='fake_A_sample')
    fake_B_sample = tf.placeholder(tf.float32, [None] + TFLAGS['input_shape'], name="fake_B_sample")

    noise_A = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="noise_A")
    noise_B = tf.placeholder(tf.float32, [None, TFLAGS['z_len']], name="noise_B")
    label_A = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="label_A")
    label_B = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="label_B")
    # c label is for real samples
    c_label = tf.placeholder(tf.float32, [None, TFLAGS['c_len']], name="c_label")

    # control variables
    real_cls_weight = tf.placeholder(tf.float32, [], name="real_cls_weight")
    fake_cls_weight = tf.placeholder(tf.float32, [], name="fake_cls_weight")
    adv_weight = tf.placeholder(tf.float32, [], name="adv_weight")
    lr = tf.placeholder(tf.float32, [], name="lr")

    inter_sum_op = []

    ### build graph
    """
    fake_A = GenA(noise_A); GenA.set_reuse() # GenA and GenB only used once
    fake_B = GenB(noise_B); GenB.set_reuse()

    # transF and TransB used three times
    trans_real_A = TransF(real_A); TransF.set_reuse() # domain B
    trans_fake_A = TransF(fake_A) # domain B
    trans_real_B = TransB(real_B); TransB.set_reuse() # domain A
    trans_fake_B = TransB(fake_B) # domain A
    rec_real_A = TransB(trans_real_A)
    rec_fake_A = TransB(trans_fake_A)
    rec_real_B = TransF(trans_real_B)
    rec_fake_B = TransF(trans_fake_B)

    # DiscA and DiscB reused many times
    disc_real_A, cls_real_A = DiscA(x_real)[:2]; DiscA.set_reuse() 
    disc_fake_A, cls_fake_A = DiscA(x_fake)[:2]; 
    disc_trans_real_B, cls_trans_real_B = DiscA(trans_real_B)[:2]
    disc_trans_fake_B, cls_trans_fake_B = DiscA(trans_real_B)[:2]
    disc_rec_real_A, cls_rec_real_A = DiscA(rec_real_A)[:2]
    disc_rec_fake_A, cls_rec_fake_A = DiscA(rec_real_A)[:2]

    disc_real_B, cls_real_B = DiscB(x_real)[:2]; DiscB.set_reuse() 
    disc_fake_B, cls_fake_B = DiscB(x_fake)[:2]; 
    disc_trans_real_A, cls_trans_real_A = DiscB(trans_real_A)[:2]
    disc_trans_fake_A, cls_trans_fake_A = DiscB(trans_real_A)[:2]
    disc_rec_real_B, cls_rec_real_B = DiscB(rec_real_B)[:2]
    disc_rec_fake_B, cls_rec_fake_B = DiscB(rec_real_B)[:2]
    """
    
    side_noise_A = tf.concat([noise_A, label_A], axis=1, name='side_noise_A')
    side_noise_B = tf.concat([noise_B, label_B], axis=1, name='side_noise_B')

    trans_real_A = TransF(real_A_sample); TransF.set_reuse() # domain B
    trans_real_B = TransB(real_B_sample); TransB.set_reuse() # domain A
    TransF.trans_real_A = trans_real_A
    TransB.trans_real_B = trans_real_B
    rec_real_A = TransB(trans_real_A)
    rec_real_B = TransF(trans_real_B)

    # start fake building
    if TFLAGS['side_noise']:
        TransF.side_noise = side_noise_A
        TransB.side_noise = side_noise_B
        trans_fake_A = TransF(real_A_sample)
        trans_fake_B = TransB(real_B_sample)

    DiscA.fake_sample = fake_A_sample; DiscB.fake_sample = fake_B_sample
    disc_fake_A_sample, cls_fake_A_sample = DiscA(fake_A_sample)[:2]; DiscA.set_reuse()
    disc_fake_B_sample, cls_fake_B_sample = DiscB(fake_B_sample)[:2]; DiscB.set_reuse()
    disc_real_A_sample, cls_real_A_sample = DiscA(real_A_sample)[:2]
    disc_real_B_sample, cls_real_B_sample = DiscB(real_B_sample)[:2]
    disc_trans_real_A, cls_trans_real_A = DiscB(trans_real_A)[:2]
    disc_trans_real_B, cls_trans_real_B = DiscA(trans_real_B)[:2]
    
    if TFLAGS['side_noise']:
        disc_trans_fake_A, cls_trans_fake_A = DiscB(trans_real_A)[:2]
        disc_trans_fake_B, cls_trans_fake_B = DiscA(trans_real_B)[:2]
        

    def disc_loss(disc_fake_out, disc_real_out, disc_model, adv_weight=1.0, name="NaiveDisc", acc=True):
        softL_c = 0.05
        with tf.name_scope(name):
            raw_disc_cost_real = tf.reduce_mean(tf.square(
                disc_real_out - tf.ones_like(disc_real_out) * np.abs(np.random.normal(1.0, softL_c))),
                name="raw_disc_cost_real")

            raw_disc_cost_fake = tf.reduce_mean(tf.square(
                disc_fake_out - tf.zeros_like(disc_fake_out)),
                name="raw_disc_cost_fake")

            disc_cost = tf.multiply(adv_weight,
                (raw_disc_cost_fake + raw_disc_cost_real)/2, name="disc_cost")

            disc_fake_sum = [
                tf.summary.scalar("DiscFakeRaw", raw_disc_cost_fake),
                tf.summary.scalar("DiscRealRaw", raw_disc_cost_real)]
            
            if acc:
                disc_model.cost += disc_cost
                disc_model.sum_op.extend(disc_fake_sum)
            else:
                return disc_cost, disc_fake_sum

    def gen_loss(disc_fake_out, gen_model, adv_weight=1.0, name="Naive", acc=True):
        softL_c = 0.05
        with tf.name_scope(name):
            raw_gen_cost = tf.reduce_mean(tf.square(
                disc_fake_out - tf.ones_like(disc_fake_out) * np.abs(np.random.normal(1.0, softL_c))),
                name="raw_gen_cost")

            gen_cost = tf.multiply(raw_gen_cost, adv_weight, name="gen_cost")

        if acc:
            gen_model.cost += gen_cost
        else:
            return gen_cost, []
    
    disc_loss(disc_fake_A_sample, disc_real_A_sample, DiscA, adv_weight=TFLAGS['adv_weight'], name="DiscA_Loss")
    disc_loss(disc_fake_B_sample, disc_real_B_sample, DiscB, adv_weight=TFLAGS['adv_weight'], name="DiscA_Loss")
    
    gen_loss(disc_trans_real_A, TransF, adv_weight=TFLAGS['adv_weight'], name="NaiveGenA")
    gen_loss(disc_trans_real_B, TransB, adv_weight=TFLAGS['adv_weight'], name="NaiveTransA")

    if TFLAGS['side_noise']:
        # extra loss is for noise not equal to zero
        TransF.extra_loss = TransB.extra_loss = 0
        TransF.extra_loss = tf.identity(TransF.cost)
        TransB.extra_loss = tf.identity(TransB.cost)

        cost_, _ = gen_loss(disc_trans_fake_A, TransF, adv_weight=TFLAGS['adv_weight'],
            name="NaiveGenAFake", acc=False)
        TransF.extra_loss += cost_
        cost_, _ = gen_loss(disc_trans_fake_B, TransB, adv_weight=TFLAGS['adv_weight'],
            name="NaiveGenBFake", acc=False)
        TransB.extra_loss += cost_

    #GANLoss(disc_rec_A, DiscA, TransB, adv_weight=TFLAGS['adv_weight'], name="NaiveGenA")
    #GANLoss(disc_rec_B, DiscA, TransF, adv_weight=TFLAGS['adv_weight'], name="NaiveTransA")

    # cycle consistent loss
    def cycle_loss(trans, origin, weight=10.0, name="cycle"):
        with tf.name_scope(name):
            # using gray
            trans = tf.reduce_mean(trans, axis=[3])
            origin = tf.reduce_mean(origin, axis=[3])
            cost_ = tf.reduce_mean(tf.abs(trans - origin)) * weight
            sum_ = tf.summary.scalar("Rec", cost_)
        
        return cost_, sum_
    
    cost_, sum_ = cycle_loss(rec_real_A, real_A_sample, TFLAGS['AE_weight'], name="cycleA")
    TransF.cost += cost_; TransB.cost += cost_; TransF.sum_op.append(sum_)

    cost_, sum_ = cycle_loss(rec_real_B, real_B_sample, TFLAGS['AE_weight'], name="cycleB")
    TransF.cost += cost_; TransB.cost += cost_; TransB.sum_op.append(sum_)

    clsB_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=cls_real_B_sample,
        labels=c_label), name="clsB_real")
        
    if TFLAGS['side_noise']:
        clsB_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=cls_trans_fake_A,
            labels=c_label), name="clsB_fake")

    DiscB.cost += clsB_real * real_cls_weight
    DiscB.sum_op.extend([
            tf.summary.scalar("RealCls", clsB_real)])
    
    if TFLAGS['side_noise']:
        DiscB.extra_loss = clsB_fake * fake_cls_weight
        TransF.extra_loss += clsB_fake * fake_cls_weight
       
        # extra loss for integrate stochastic
        cost_, sum_ = cycle_loss(rec_real_A, real_A_sample, TFLAGS['AE_weight'], name="cycleADisturb")
        TransF.extra_loss += cost_; TransF.sum_op.append(sum_)
        TransF.extra_loss += cls_trans_real_A *  fake_cls_weight

    # add interval summary
    edge_num = int(np.sqrt(TFLAGS['batch_size']))
    if edge_num > 4:
        edge_num = 4

    grid_real_A = ops.get_grid_image_summary(real_A_sample, edge_num)
    inter_sum_op.append(tf.summary.image("Real A", grid_real_A))
    grid_real_B = ops.get_grid_image_summary(real_B_sample, edge_num)
    inter_sum_op.append(tf.summary.image("Real B", grid_real_B))

    grid_trans_A = ops.get_grid_image_summary(trans_real_A, edge_num)
    inter_sum_op.append(tf.summary.image("Trans A", grid_trans_A))
    grid_trans_B = ops.get_grid_image_summary(trans_real_B, edge_num)
    inter_sum_op.append(tf.summary.image("Trans B", grid_trans_B))

    if TFLAGS['side_noise']:
        grid_fake_A = ops.get_grid_image_summary(trans_fake_A, edge_num)
        inter_sum_op.append(tf.summary.image("Fake A", grid_fake_A))
        grid_fake_B = ops.get_grid_image_summary(trans_fake_B, edge_num)
        inter_sum_op.append(tf.summary.image("Fake B", grid_fake_B))

    # merge summary op
    for m, n in zip(models, model_names):
        m.cost = tf.identity(m.cost, "Total" + n)
        m.sum_op.append(tf.summary.scalar("Total" + n, m.cost))
        m.sum_op = tf.summary.merge(m.sum_op)
        m.get_trainable_variables()

    inter_sum_op = tf.summary.merge(inter_sum_op)

    # Not applying update op will result in failure
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        for m, n in zip(models, model_names):
            m.train_op = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.5,
                beta2=0.9).minimize(m.cost, var_list=m.vars)
            
            if m.extra_loss is not 0:
                m.extra_train_op = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.5,
                beta2=0.9).minimize(m.extra_loss, var_list=m.vars)
            
            print("=> ##### %s Variable #####" % n)
            m.print_trainble_vairables()

    print("=> ##### All Variable #####")
    for v in tf.trainable_variables():
        print("%s" % v.name)

    ctrl_weight = {
            "real_cls"  : real_cls_weight,
            "fake_cls"  : fake_cls_weight,
            "adv"       : adv_weight,
            "lr"        : lr
        }

    # basic settings
    trainer_feed = {
        "ctrl_weight": ctrl_weight,
        "dataloader": dl,
        "int_sum_op": inter_sum_op,
        "gpu_mem": TFLAGS['gpu_mem'],
        "FLAGS": FLAGS,
        "TFLAGS": TFLAGS
    }

    for m, n in zip(models, model_names):
        trainer_feed.update({n:m}) 

    Trainer = trainer.cycle_trainer.CycleTrainer
    if TFLAGS['side_noise']:
        Trainer.noises = [noise_A, noise_B]
        Trainer.labels = [label_A, label_B]
    # input
    trainer_feed.update({
        "inputs": [real_A_sample, real_B_sample, c_label]
    })
    
    ModelTrainer = Trainer(**trainer_feed)

    command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    command_controller.start_thread()

    ModelTrainer.init_training()

    DISC_PATH = [
        'success/goodmodel_dragan_sketch2_disc.npz', 'success/goodmodel_dragan_anime1_disc.npz']
    
    DiscA.load_from_npz(DISC_PATH[0], ModelTrainer.sess)
    DiscB.load_from_npz(DISC_PATH[1], ModelTrainer.sess)

    ModelTrainer.train()

if __name__ == "__main__":
    main()
