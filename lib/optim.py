import tensorflow as tf

import time
import numpy as np
from scipy import misc
import os
import sys
sys.path.insert(0, ".")
import pprint

# model
import model, loss
from lib import files, utils, dataloader, ops

from skimage import io, transform

class PictureOptimizer(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.model_dir = CONFIG['model_dir'] + CONFIG['model_name'] + CONFIG['sery']
        self.using_cgan = CONFIG['cgan']

        # model_class = model.simple_generator.SimpleConvolutionGenerator
        model_class = getattr(getattr(model, CONFIG['model_class']), CONFIG['gen_model'])
        self.gen_model = model_class(**CONFIG['gen_config'])
        model_class = getattr(getattr(model, CONFIG['model_class']), CONFIG['disc_model'])
        self.disc_model = model_class(**CONFIG['disc_config'])

        # the output image size
        self.size = (2 ** CONFIG['n_layers']) * CONFIG['map_size']
        self.input_dim = CONFIG['gen_config']['out_dim']
        self.n_attrs = CONFIG['disc_config']['n_attrs']
        self.output_shape = [self.size, self.size, self.input_dim]
        self.delta =  1. - (1. / (1. + np.exp(-5.)) - 1. / (1. + np.exp(5.)))

        # init
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=self.config)
        self.load_model()
        self.build_graph()
        print("=> init over")

    def load_model(self):
        self.raw_z_noise = tf.placeholder(tf.float32, [None, 128], name="z_noise")
        self.z_noise = tf.tanh(self.raw_noise)
        self.x_real = tf.placeholder(tf.float32, [None] + self.output_shape, name="x_real")
        self.fake_sample = tf.placeholder(tf.float32, [None] + self.output_shape, name="fake_sample")
        
        if self.using_cgan:
            self.raw_c_noise = tf.placeholder(tf.float32, [None, self.n_attrs], name="z_noise")
            self.c_noise = tf.sigmoid(self.raw_noise)
            self.c_label = tf.placeholder(tf.float32, [None, self.n_attrs], name="c_label")
            
            self.x_fake = self.gen_model([self.z_noise, self.c_noise])
        else:
            self.x_fake = self.gen_model(self.z_noise)

        # build model
        self.disc_fake = self.disc_model(self.fake_sample)[0]
        self.gen_model.load_from_npz(self.model_dir + "gen.npz", self.sess)
        self.disc_model.load_from_npz(self.model_dir + "disc.npz", self.sess)

        self.c_noise_fixed = False

        self.sketch = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.mask = tf.placeholder(tf.float32, [None, 128, 128, 3])

        self.gen_output = self.x_fake

        self.feed = {
            self.gen_model.training: False,
            self.disc_model.training: False,
            self.gen_model.keep_prob: 1.0,
            self.disc_model.keep_prob: 1.0
        }

        self.feed_gradient = {
            self.gen_model.training: False,
            self.disc_model.training: False,
            self.gen_model.keep_prob: 1.0,
            self.disc_model.keep_prob: 1.0
        }

    def build_graph(self):
        print("=> building graph")
        out_255 = (self.gen_output + 1.) * 127.5
        x = tf.multiply((out_255 - self.sketch), self.mask)
        self.mse_batch = tf.reduce_sum(tf.abs(x), axis=[1, 2, 3]) / (tf.reduce_sum(self.mask, axis=[1, 2, 3]) + 1.)
        self.mse_loss = tf.reduce_sum(tf.abs(self.x)) / (tf.reduce_sum(self.mask) + 1e-6)

        self.gz = tf.gradients(self.mse_loss, [self.raw_z_noise])[0]
        if self.using_cgan:
            self.gc = tf.gradients(self.mse_loss, [self.raw_c_noise])[0]

    def get_c_noise(self):
        if self.using_cgan == False:
            return
        # raw c noise is in [-2, 2]
        self.origin_raw_c = ops.random_boolean((1, self.n_attrs), if_getchu=True) * 4 - 2

    def change_c_noise(self, lr):
        if self.using_cgan  == True:
            self.origin_raw_c += -lr * self.gradc[0]
            gradc_std = np.std(self.gradc[0])
            self.origin_c_noise = self.sigmoid_arc(self.arc_origin_c_noise)
            print("=> Grad magnitude: [%.4f, %.4f]" % (self.gradc[0].min(), self.gradc[0].max()))

    def get_z_noise(self):
        # origin raw z noise: [-2, 2]
        self.origin_raw_z = np.random.rand((1, 128)) * 4 - 2

    def change_z_noise(self, lr):
        self.origin_raw_z += -lr * self.gradz[0]
        print("=> Grad magnitude: [%.4f, %.4f]" % (self.gradz[0].min(), self.gradz[0].max()))

    def get_noise(self):
        self.get_z_noise()
        self.get_c_noise()

    def change_noise(self, lr_z, lr_c):
        self.change_z_noise(lr_z)
        self.change_c_noise(lr_c)

    def get_noise_batch(self, raw_z, lr, grad):
        print("=> noise: [%f, %f]" % (raw_z.min(), raw_z.max()))
        grad_std = np.std(grad)

        noise_list = [raw_z - lr * grad]

        for i in range(3):
            noise_list.append(raw_z - (2 ** (i+1) ) * lr * grad)
        for i in range(4):
            noise_list.append(raw_z - (2 ** i) * lr * grad + lr * ops.random_normal(raw_z.shape) * grad_std * 0.3)
        
        noise_list = np.concatenate(noise_list, axis=0)
        noise_list = np.maximum(noise_list, -5)
        noise_list = np.minimum(noise_list, 5)
        return noise_list

    def change_noise_batch(self, lr_z, lr_c):
        new_z_list = self.get_noise_batch(self.origin_raw_z, lr_z, self.gradz[0])
        self.feed.update({self.raw_z_noise: new_z_list})
        if self.using_cgan:
            new_c_list = self.get_noise_batch(self.origin_raw_c, lr_c, self.gradc[0])
            self.feed.update({self.raw_c_noise : new_c_list})
        
        self.output = self.sess.run(self.gen_output, self.feed)

        self.feed.update({
            self.fake_sample: self.output,
            self.sketch : self.feed_gradient[self.sketch],
            self.mask : self.feed_gradient[self.mask]})

        self.disc_val, self.mse_val = self.sess.run([self.disc_fake, self.mse_batch], self.feed)
        delta = 0.04 * (self.mse_val - self.loss_)
        self.disc_val = self.disc_val[:, 0]
        print("=> Disc val:" + str(self.disc_val))
        print("=> Delta:" + str(delta))
        self.disc_val -= delta
        ind = np.argmax(self.disc_val)
        print("=> Best index: %d" % ind)

        self.origin_raw_z = new_z_list[ind:ind+1]
        if self.using_cgan:
            self.origin_raw_c = new_c_list[ind:ind+1]
            
        self.output = np.array(self.trans(self.output[0]))

    def get_output(self):
        self.feed.update({self.raw_z_noise: self.origin_raw_z})
        if self.using_cgan == True:
            self.feed.update({self.raw_c_noise : self.origin_raw_c})

        self.output = self.sess.run(self.gen_output, self.feed)

        self.feed.update({self.fake_sample:self.output})
        self.disc_val = self.sess.run([self.disc_fake], self.feed)[0]
        print("=> Disc val %f" % self.disc_val)
        
        if self.disc_val < -5:
            print("=> Bad sample, regenerate " + (str)(self.disc_val))
            return False
        else:
            self.output = np.array(self.trans(self.output[0]))
            return True

    def save_variables(self):
        self.save_z_noise = self.origin_raw_z

        if self.using_cgan == True:
            self.save_c_noise = self.origin_raw_c

    def recover_variables(self):
        self.origin_raw_z = self.save_z_noise
        if self.using_cgan == True:
            self.origin_raw_c = self.save_c_noise

    def get_gradient(self, sketch_img, mask_img):
        self.feed_gradient.update({
            self.raw_z_noise: self.origin_raw_z,
            self.sketch : sketch_img,
            self.mask : mask_img
        })

        if self.using_cgan == True:
            self.feed_gradient.update({self.raw_c_noise : self.origin_raw_c})
        
        if self.using_cgan:
            self.loss_, self.gradz, self.gradc = self.sess.run([self.mse_loss, self.gz, self.gc], self.feed_gradient)
        else:
            self.loss_, self.gradz = self.sess.run([self.mse_loss, self.gz], self.feed_gradient)

    def generate(self, sketch_img, mask_img, raw_z, raw_c=[], file_lr_path = "learning_rate.txt"):
        print("=> loading sketch and mask")
        sketch_img = transform.resize(sketch_img, (128, 128))[:, :, :3]
        mask_img = transform.resize(mask_img, (128, 128))[:, :, :3]
        sketch_img = sketch_img.astype(np.float32).reshape(1, 128, 128, 3)
        mask_img = mask_img.astype(np.float32).reshape(1, 128, 128, 3)

        # convert sketch to 255 scale
        if sketch_img.max() < 1.1:
            sketch_img = sketch_img * 255.
        # convert mask to 1 scale
        if mask_img.max() > 1.1:
            mask_img = mask_img / 255.

        print("=> Sketch: [%f, %f]" % (sketch_img.min(), sketch_img.max()))
        print("=> Mask: [%f, %f]" % (mask_img.min(), mask_img.max()))

        self.origin_raw_z = raw_z
        self.origin_raw_c = raw_c

        with open(file_lr_path) as f:
            lr_z, lr_c = f.read().split('\n')
            lr_z = float(lr_z)
            lr_c = float(lr_c)
        print("=> lr_z: %f\t\tlr_c: %f" % (lr_z, lr_c))

        time_start = time.time()
        
        self.get_gradient(sketch_img, mask_img)
        self.save_variables()
        self.change_noise_batch(lr_z, lr_c)

        time_end=time.time()

        print("=> loss: %f" % self.loss_)
        print("=> Changed picture.  Time:"+(str)(time_end - time_start) + "s")

        if self.using_cgan == True:
            return self.output, self.origin_z_noise, self.arc_origin_c_noise
        else:
            return self.output, self.origin_z_noise, []

    def generate_origin(self):
        time_start = time.time()
        while(True):
            self.get_noise()
            if self.get_output() == True:
                break
        time_end=time.time()
        print ("Generated origin picture.  Time:"+(str)(time_end - time_start) + "s")
        if self.using_cgan == True:
            return self.output, self.origin_z_noise, self.arc_origin_c_noise
        else:
            return self.output, self.origin_z_noise, []

class PictureOptimizerS(object):
    def __init__(self):
        self.optimizers = []
    
    def create_new_optimizer(self, CONFIG):
        new_optimizer = PictureOptimizer(CONFIG)
        self.optimizers.append(new_optimizer)
    
    def generate(self, CONFIG, sketch, mask, z, c, file_lr_path):
        for optimizer in self.optimizers:
            if optimizer.CONFIG == CONFIG:
                return optimizer.generate(sketch, mask, z, c, file_lr_path="learning_rate.txt")
    
    def generate_origin(self, CONFIG):
        for optimizer in self.optimizers:
            if optimizer.CONFIG == CONFIG:
                return optimizer.generate_origin()