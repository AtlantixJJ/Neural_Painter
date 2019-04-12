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
tf.app.flags.DEFINE_string("train_dir", "logs/simple_getchu", "log dir")

# ----- model type flags ------ #

tf.app.flags.DEFINE_boolean("cgan", True, "If to use ACGAN")
tf.app.flags.DEFINE_integer("img_size", 128, "The size of input image, 64 | 128")
tf.app.flags.DEFINE_string("model_name", "simple", "model type: simple | simple_mask | hg | hg_mask")
tf.app.flags.DEFINE_string("data_dir", "/home/atlantix/data/celeba/img_align_celeba.zip", "data path")
tf.app.flags.DEFINE_boolean("cbn_project", True, "If to project to depth dim")
tf.app.flags.DEFINE_integer("sn", 1, "0 for no spectral norm| 1 for noise spectral norm | 2 for data spectral norm")

# ------ train control flags ----- #

tf.app.flags.DEFINE_boolean("use_cache", False, "If to use cache to prevent cactastrophic forgetting.")
tf.app.flags.DEFINE_string("gpu", "4", "which gpu to use")
tf.app.flags.DEFINE_float("g_lr", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("d_lr", 4e-4, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "training batch size")
tf.app.flags.DEFINE_integer("num_iter", 200000, "training iteration")
tf.app.flags.DEFINE_integer("dec_iter", 100000, "training iteration")
tf.app.flags.DEFINE_integer("disc_iter", 1, "discriminator training iter")
tf.app.flags.DEFINE_integer("gen_iter", 1, "generative training iter")

FLAGS = tf.app.flags.FLAGS
TFLAGS = {}

NUM_GPU = len(FLAGS.gpu.split(","))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

def main():
    size = FLAGS.img_size

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
    gen_model.cbn_project = FLAGS.cbn_project
    gen_model.spectral_norm = FLAGS.sn
    disc_model.cbn_project = FLAGS.cbn_project
    disc_model.spectral_norm = FLAGS.sn

    ModelTrainer = trainer.base_gantrainer.BaseGANTrainer(
        step_sum_op=None,
        int_sum_op=None,
        dataloader=dl,
        FLAGS=FLAGS,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_input=gen_input,
        x_real=x_real,
        label=c_label)

    g_tower_grads = []
    d_tower_grads = []
    g_optim = tf.train.AdamOptimizer(
                learning_rate=ModelTrainer.g_lr,
                beta1=0.,
                beta2=0.9)
    d_optim = tf.train.AdamOptimizer(
                learning_rate=ModelTrainer.d_lr,
                beta1=0.,
                beta2=0.9)

    grad_x = []
    grad_x_name = []
    xs = []
    x_name = []

    def tower(gpu_id, gen_input, x_real, c_label=None, c_noise=None, update_collection=None, loss_collection=[]):
        """
        The loss function builder of gen and disc
        """
        gen_model.cost = disc_model.cost = 0

        gen_model.set_phase("gpu%d" % gpu_id)
        x_fake = gen_model(gen_input, update_collection=update_collection)
        gen_model.set_reuse()
        gen_model.x_fake = x_fake

        disc_model.set_phase("gpu%d" % gpu_id)
        disc_real, real_cls_logits = disc_model(x_real, update_collection=update_collection)
        disc_model.set_reuse()
        disc_model.recorded_tensors = []
        disc_model.recorded_names = []
        disc_fake, fake_cls_logits = disc_model(x_fake, update_collection=update_collection)
        disc_model.disc_real        = disc_real       
        disc_model.disc_fake        = disc_fake       
        disc_model.real_cls_logits = real_cls_logits
        disc_model.fake_cls_logits = fake_cls_logits

        if FLAGS.cgan:
            fake_cls_cost, real_cls_cost = loss.classifier_loss(
                gen_model, disc_model, x_real, c_label, c_noise,
                weight=1.0/dataset.class_num, summary=False)

        raw_gen_cost, raw_disc_real, raw_disc_fake = loss.hinge_loss(
            gen_model, disc_model,
            adv_weight=1.0, summary=False)

        gen_model.vars = [v for v in tf.trainable_variables() if gen_model.name in v.name]
        disc_model.vars = [v for v in tf.trainable_variables() if disc_model.name in v.name]
        g_grads = tf.gradients(gen_model.cost, gen_model.vars, colocate_gradients_with_ops=True)
        d_grads = tf.gradients(disc_model.cost, disc_model.vars, colocate_gradients_with_ops=True)
        g_grads = [tf.check_numerics(g, "G grad nan: " + str(g)) for g in g_grads]
        d_grads = [tf.check_numerics(g, "D grad nan: " + str(g)) for g in d_grads]
        g_tower_grads.append(g_grads)
        d_tower_grads.append(d_grads)

        tensors = gen_model.recorded_tensors + disc_model.recorded_tensors
        names = gen_model.recorded_names + disc_model.recorded_names
        if gpu_id == 0: x_name.extend(names)
        xs.append(tensors)
        names = names[::-1]
        tensors = tensors[::-1]
        grads = tf.gradients(disc_fake, tensors, colocate_gradients_with_ops=True)
        for n,g in zip(names, grads):
            print(n, g)
        grad_x.append([tf.check_numerics(g, "BP nan: " + str(g)) for g in grads])
        if gpu_id == 0: grad_x_name.extend(names)
        disc_model.recorded_tensors = []
        disc_model.recorded_names = []
        gen_model.recorded_tensors = []
        gen_model.recorded_names = []

        return gen_model.cost, disc_model.cost, [fake_cls_cost, real_cls_cost, raw_gen_cost, raw_disc_real, raw_disc_fake]

    def average_gradients(tower_grads):
        average_grads = []
        num_gpus = len(tower_grads)
        num_items = len(tower_grads[0])
        for i in range(num_items):
            average_grads.append(0.0)
            for j in range(num_gpus):
                average_grads[i] += tower_grads[j][i]
            average_grads[i] /= num_gpus
        return average_grads

    sbs = FLAGS.batch_size // NUM_GPU
    for i in range(NUM_GPU):
        if i == 0:
            update_collection = None
        else:
            update_collection = "no_ops"

        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            if FLAGS.cgan:
                l1, l2, ot1 = tower(i, 
                    [z_noise[sbs*i:sbs*(i+1)], c_noise[sbs*i:sbs*(i+1)]],
                    x_real[sbs*i:sbs*(i+1)],
                    c_label[sbs*i:sbs*(i+1)],
                    c_noise[sbs*i:sbs*(i+1)],
                    update_collection=update_collection)
            else:
                l1, l2, ot1 = tower(i,
                    z_noise[sbs*i:sbs*(i+1)],x_real[sbs*i:sbs*(i+1)],
                    update_collection=update_collection)

        if i == 0:
            int_sum_op = []

            grid_x_fake = ops.get_grid_image_summary(gen_model.x_fake, 4)
            int_sum_op.append(tf.summary.image("generated image", grid_x_fake))

            grid_x_real = ops.get_grid_image_summary(x_real, 4)
            int_sum_op.append(tf.summary.image("real image", grid_x_real))

            step_sum_op = []
            sub_loss_names = ["fake_cls", "real_cls", "gen", "disc_real", "disc_fake"]
            for n,l in zip(sub_loss_names, ot1):
                step_sum_op.append(tf.summary.scalar(n, l))
            step_sum_op.append(tf.summary.scalar("gen", gen_model.cost))
            step_sum_op.append(tf.summary.scalar("disc", disc_model.cost))

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        g_grads = average_gradients(g_tower_grads)
        d_grads = average_gradients(d_tower_grads)

        gen_model.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=gen_model.name + "/")
        disc_model.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=disc_model.name + "/")
        
        print(gen_model.update_ops)
        print(disc_model.update_ops)

        def merge_list(l1, l2):
            return [[l1[i], l2[i]] for i in range(len(l1))]

        with tf.control_dependencies(gen_model.update_ops):
            gen_model.train_op = g_optim.apply_gradients(merge_list(g_grads, gen_model.vars))
        with tf.control_dependencies(disc_model.update_ops):
            disc_model.train_op = d_optim.apply_gradients(merge_list(d_grads, disc_model.vars))
    
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

    ModelTrainer.int_sum_op = tf.summary.merge(int_sum_op)
    ModelTrainer.step_sum_op = tf.summary.merge(step_sum_op)
    ModelTrainer.grad_x = grad_x
    ModelTrainer.grad_x_name = grad_x_name
    ModelTrainer.xs = xs
    ModelTrainer.x_name = x_name
    #command_controller = trainer.cmd_ctrl.CMDControl(ModelTrainer)
    #command_controller.start_thread()

    print("=> Build train op")
    # ModelTrainer.build_train_op()
    
    print("=> ##### Generator Variable #####")
    gen_model.print_variables()
    print("=> ##### Discriminator Variable #####")
    disc_model.print_variables()
    
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
