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
    def __init__(self, CONFIG, sess):
        self.sess = sess

        # The computation accuracy and the function domain
        self.EPS = 1e-6
        self.RANGE = np.arctanh(1 - self.EPS / 3.0)

        self.CONFIG = CONFIG

        # determine gen & disc model path
        if len(CONFIG["sery"]) > 0:
            self.disc_dir = CONFIG['model_dir'] + ( "gen_%s.npz" % CONFIG["sery"]) 
            self.gen_dir  = CONFIG['model_dir'] + ("disc_%s.npz" % CONFIG["sery"]) 
        else:
            self.disc_dir = CONFIG['model_dir'] +  "gen.npz" 
            self.gen_dir  = CONFIG['model_dir'] + "disc.npz" 

        self.using_cgan = CONFIG['cgan']
        
        # default model class:
        # model_class = model.simple.SimpleConvolutionGenerator
        model_class = getattr(getattr(model, CONFIG['model_class']), CONFIG['gen_model'])
        gen_config = CONFIG['gen_config']
        self.gen_model = model_class(**gen_config)
        model_class = getattr(getattr(model, CONFIG['model_class']), CONFIG['disc_model'])
        disc_config = CONFIG['disc_config']
        self.disc_model = model_class(**disc_config)

        # the output image size
        self.size = (2 ** gen_config['n_layer']) * gen_config['map_size']
        self.input_dim = gen_config['out_dim']
        self.n_attrs = disc_config['n_attr']
        self.output_shape = [self.size, self.size, self.input_dim]
        self.trans = ops.get_inverse_process_fn(kind='tanh')

        # init
        self.load_model()
        self.build_graph()
        
        print("=> init over")

    def load_model(self):
        # [-5, +5]
        self.raw_z_noise = tf.placeholder(tf.float32, [None, 128], name="z_noise")
        # We assume noise to be in a gaussian distribution with sigma=1
        # Most of the probability density focus in 3 sigma region
        self.z_noise = tf.tanh(self.raw_z_noise) * 3
        self.x_real = tf.placeholder(tf.float32, [None] + self.output_shape, name="x_real")
        self.fake_sample = tf.placeholder(tf.float32, [None] + self.output_shape, name="fake_sample")
        
        if self.using_cgan:
            self.raw_c_noise = tf.placeholder(tf.float32, [None, self.n_attrs], name="z_noise")
            self.c_noise = tf.sigmoid(self.raw_c_noise)
            self.c_label = tf.placeholder(tf.float32, [None, self.n_attrs], name="c_label")
            
            self.x_fake = self.gen_model([self.z_noise, self.c_noise], "no_ops")
        else:
            self.x_fake = self.gen_model(self.z_noise, "no_ops")

        self.disc_fake = self.disc_model(self.fake_sample, "no_ops")[0]

        self.sess.run(tf.global_variables_initializer())

        self.gen_model.load_from_npz(self.gen_dir, self.sess)
        self.disc_model.load_from_npz(self.disc_dir, self.sess)

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
        # output in [0, 255] scale
        out_255 = (self.gen_output + 1.) * 127.5
        x = tf.multiply((out_255 - self.sketch), self.mask)
        # all the difference in a batch
        self.mse_loss = tf.reduce_sum(tf.abs(x)) / (tf.reduce_sum(self.mask) + 1.0)
        # loss of each instance
        self.mse_batch = tf.reduce_sum(tf.abs(x), axis=[1, 2, 3]) / (tf.reduce_sum(self.mask, axis=[1, 2, 3]) + 1.)
        
        self.gz = tf.gradients(self.mse_loss, [self.raw_z_noise])[0]
        if self.using_cgan:
            self.gc = tf.gradients(self.mse_loss, [self.raw_c_noise])[0]

    def get_c_noise(self):
        if self.using_cgan == False:return
        # raw c noise is gaussian
        self.origin_raw_c = ops.get_random("normal", (1, self.n_attrs))

    def change_c_noise(self, lr):
        if self.using_cgan  == False: return
        self.origin_raw_c += -lr * self.gradc[0]
        noise_c = tf.sigmoid(tf.constant(self.origin_raw_c)).eval()
        print("=> c vector: min=%.4f, max=%.4f, norm=%.4f" % (
            noise_c.min(), noise_c.max(), np.linalg.norm(noise_c, 2)))

    def get_z_noise(self):
        # origin raw z noise: guassian with 3 sigma
        self.origin_raw_z = ops.get_random("normal", (1, 128)) * 3

    def change_z_noise(self, lr):
        self.origin_raw_z += -lr * self.gradz[0]
        noise_z = np.tanh(self.origin_raw_z)
        print("=> z vector: min=%.4f, max=%.4f, norm=%.4f" % (
            noise_z.min(), noise_z.max(), np.linalg.norm(noise_z, 2)))

    def get_noise(self):
        self.get_z_noise()
        self.get_c_noise()

    def change_noise(self):
        self.change_z_noise(self.lr_z)
        self.change_c_noise(self.lr_c)

    def get_noise_batch(self, raw_z, lr, grad):
        print("=> noise: min=%f, max=%f" % (raw_z.min(), raw_z.max()))
        grad_std = np.std(grad)

        noise_list = [raw_z - lr * grad]

        for i in range(3):
            noise_list.append(raw_z - (2 ** (i+1) ) * lr * grad)
        for i in range(4):
            noise_list.append(raw_z - (2 ** i) * lr * grad + lr * ops.get_random("normal", raw_z.shape) * grad_std * 0.1)
        
        noise_list = np.concatenate(noise_list, axis=0)
        noise_list = np.maximum(noise_list, -self.RANGE)
        noise_list = np.minimum(noise_list, self.RANGE)
        return noise_list

    def change_noise_batch(self):
        with open("learning_rate.txt", "r") as f:
            self.lr_z = float(f.readline().strip())
            self.lr_c = float(f.readline().strip())
        new_z_list = self.get_noise_batch(self.origin_raw_z, self.lr_z, self.gradz[0])
        self.feed.update({self.raw_z_noise: new_z_list})
        if self.using_cgan:
            new_c_list = self.get_noise_batch(self.origin_raw_c, self.lr_c, self.gradc[0])
            self.feed.update({self.raw_c_noise : new_c_list})
        
        self.output = self.sess.run(self.gen_output, self.feed)

        self.feed.update({
            self.fake_sample: self.output,
            self.sketch : self.feed_gradient[self.sketch],
            self.mask : self.feed_gradient[self.mask]})

        self.disc_val, self.mse_val = self.sess.run([self.disc_fake, self.mse_batch], self.feed)
        delta = 0.04 * (self.mse_val - self.loss_) # suppose to be negative
        self.disc_val = self.disc_val[:, 0] # the more positive, the better
        print("=> Disc val: " + str(self.disc_val))
        print("=> Feature Value: " + str(delta))
        self.disc_val -= delta
        ind = np.argmax(self.disc_val)
        print("=> Best index: %d" % ind)

        new_z = new_z_list[ind:ind+1]
        delta_z = new_z - self.origin_raw_z
        print("=> Delta z: min=%f, max=%f, norm=%f" % (
            delta_z.min(),
            delta_z.max(),
            np.linalg.norm(delta_z, 2)))
        self.origin_raw_z = new_z
        if self.using_cgan:
            new_c = new_c_list[ind:ind+1]
            delta_c = new_c - self.origin_raw_c
            print("=> Delta c: min=%f, max=%f, norm=%f" % (
                delta_c.min(),
                delta_c.max(),
                np.linalg.norm(delta_c, 2)))
            self.origin_raw_c = new_c
            
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
            print("=> Grad c min=%.4f, max=%.4f, norm=%.4f" % (
                self.gradc[0].min(),
                self.gradc[0].max(),
                np.linalg.norm(self.gradc[0], 2)))
        else:
            self.loss_, self.gradz = self.sess.run([self.mse_loss, self.gz], self.feed_gradient)
        
        print("=> Grad z min=%.4f, max=%.4f, norm=%.4f" % (
            self.gradz[0].min(),
            self.gradz[0].max(),
            np.linalg.norm(self.gradz[0], 2)))

    def generate(self, sketch_img, mask_img, raw_z, raw_c=[], file_lr_path="learning_rate.txt"):
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

        self.origin_raw_z = raw_z
        self.origin_raw_c = raw_c

        time_start = time.time()
        
        self.get_gradient(sketch_img, mask_img)
        self.save_variables()
        self.change_noise_batch()

        time_end = time.time()

        print("=> loss: %f" % self.loss_)
        print("=> Editing image: " + (str)(time_end - time_start) + "s")

        if self.using_cgan == True:
            return self.output, self.origin_raw_z, self.origin_raw_c
        else:
            return self.output, self.origin_raw_z, []

    def generate_origin(self):
        time_start = time.time()
        while(True):
            self.get_noise()
            if self.get_output() == True:
                break
        time_end = time.time()
        print ("=> Generated origin image: " + (str)(time_end - time_start) + "s")
        if self.using_cgan == True:
            return self.output, self.origin_raw_z, self.origin_raw_c
        else:
            return self.output, self.origin_raw_z, []

class PictureOptimizerS(object):
    def __init__(self, CONFIG):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu']

        #tf.enable_eager_execution()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

        self.optimizers = []
        for model_config in CONFIG['models'].values():
            self.optimizers.append(PictureOptimizer(model_config, sess))
    
    def generate(self, CONFIG, sketch, mask, z, c, file_lr_path):
        for optimizer in self.optimizers:
            if optimizer.CONFIG == CONFIG:
                return optimizer.generate(sketch, mask, z, c, file_lr_path="learning_rate.txt")
    
    def generate_origin(self, CONFIG):
        for optimizer in self.optimizers:
            if optimizer.CONFIG == CONFIG:
                return optimizer.generate_origin()

if __name__ == "__main__":
    # test
    import json
    CONFIG = {}
    with open("nim_server/config2.json", "r") as f:
        CONFIG = json.load(f)

    optimizer = PictureOptimizerS(CONFIG)