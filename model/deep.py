import tensorflow as tf

from tensorflow.contrib import layers as L
import numpy as np
import math
from model.simple import SimpleConvolutionDiscriminator, SimpleConvolutionGenerator

from lib import ops, files, layers, utils
from model import basic

class ResidualGenerator(SimpleConvolutionGenerator):
    def __init__(self, **kwargs):
        super(ResidualGenerator, self).__init__(**kwargs)
    
    def build_inference(self, input, update_collection=None):
        # conditional bn: must use with conditional GAN
        bn_partial = self.get_batchnorm()

        x = layers.linear("fc1", input, 
                (self.map_size ** 2) * self.get_depth(0),
                self.spectral_norm, self.reuse)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)
        print("=> fc1:\t" + str(x.get_shape()))
        
        for i in range(self.n_layer):
            name = "res%d" % (i+1)
            x = layers.upsample_residual_block(name, x, self.get_depth(i+1),
                tf.nn.relu, bn_partial,
                self.spectral_norm, self.reuse)
            print("=> " + name + ":\t" + str(x.get_shape()))

        x = bn_partial("out/bn", x)
        x = tf.nn.relu(x)
        x = layers.conv2d("conv1", x, self.out_dim, self.ksize, 1, self.spectral_norm, self.reuse)

        self.out = tf.nn.tanh(x)
        print("=> gen:\t" + str(self.out.get_shape()))

        return self.out

class ResidualDiscriminator(SimpleConvolutionDiscriminator):
    def __init__(self, **kwargs):
        super(ResidualDiscriminator, self).__init__(**kwargs)
        
    def build_inference(self, input, update_collection=None):
        # usually discriminator do not use bn
        bn_partial = self.get_discriminator_batchnorm()

        x = layers.conv2d("conv1", input, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, self.reuse)
        x = bn_partial("bn1", x)
        x = layers.LeakyReLU(x)
        print("=> conv1:\t" + str(x.get_shape()))

        self.mid_layers = self.n_layer // 2 + 1

        for i in range(self.n_layer):
            name = "res%d" % (i+1)
            x = layers.downsample_residual_block(name, x, self.get_depth(i+1),
                layers.LeakyReLU, bn_partial,
                self.spectral_norm, self.reuse)
            print("=> " + name + ":\t" + str(x.get_shape()))

        x = layers.LeakyReLU(x)
        h = tf.reduce_mean(x, axis=[1, 2])
        print("=> gap:\t" + str(h.get_shape()))

        self.disc_out = layers.linear("disc/fc", h, 1,
                        0, self.reuse)
        self.cls_out = layers.linear("cls/fc", h, self.n_attr,
                        0, self.reuse)

        # class conditional info
        """
        if self.label is not None:
            dim = h.get_shape()[-1]
            emb_label = layers.linear("class/emd", self.label, dim,
                self.spectral_norm, self.reuse)
            delta = tf.reduce_sum(h * emb_label, axis=[1], keepdims=True)
        """

        if self.label is not None:
            return self.disc_out, self.cls_out
        else:
            return self.disc_out, 0

class DeepGenerator(SimpleConvolutionGenerator):
    def __init__(self, n_res=2, **kwargs):
        super(DeepGenerator, self).__init__(**kwargs)
        self.n_res = n_res

    def build_inference(self, input, update_collection=None):
        # conditional bn: must use with conditional GAN
        cbn_partial = utils.partial(layers.conditional_batch_normalization, conditions=input, phase=self.phase, update_collection=update_collection, is_project=self.cbn_project, reuse=self.reuse)
        bn_partial = utils.partial(layers.default_batch_norm, phase=self.phase, update_collection=update_collection, reuse=self.reuse)

        x = layers.linear("fc1", input, 
                    self.map_size ** 2 * self.get_depth(0),
                    self.spectral_norm, update_collection, self.reuse)
        x = tf.reshape(x,
            shape=[-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial("fc1/bn", x)
        x = tf.nn.relu(x)
        print("=> fc1: " + str(x.get_shape()))

        for i in range(self.n_layer // 2):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.get_depth(i+1), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = cbn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ": " + str(x.get_shape()))

        base_x = tf.identity(x)

        res_cnt = 1
        for i in range(self.n_layer):
            x_id = tf.identity(x)
            for j in range(self.n_res):
                name = "res%d" % res_cnt
                x = layers.simple_residual_block(name, x, 3,
                    tf.nn.relu,
                    cbn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                x = cbn_partial(name + "/bn", x)
                x = tf.nn.relu(x)
                res_cnt += 1
                print("=> " + name + ": " + str(x.get_shape()))
            x = tf.add(x, x_id, name="add")
            
        x = tf.add(x, base_x, name="bridge_join")
        x = cbn_partial("bridge/bn", x)
        x = tf.nn.relu(x)

        for i in range(self.n_layer // 2, self.n_layer):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.get_depth(i+1), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = cbn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ": " + str(x.get_shape()))

        x = layers.conv2d("conv%d" % (self.n_layer + 1), x, self.out_dim, self.ksize, 1,
                    self.spectral_norm,
                    update_collection,
                    reuse=self.reuse)

        with tf.name_scope("gen_out") as nsc:
            self.out = tf.nn.tanh(x, name=nsc)

        return self.out    

class DeepDiscriminator(SimpleConvolutionDiscriminator):
    def __init__(self, n_res=2, n_block=5, **kwargs):
        super(DeepDiscriminator, self).__init__(**kwargs)

        self.n_res = n_res
        self.n_block = n_block

    def build_inference(self, input, update_collection=None):
        def bn_partial(name, x): return x

        x = layers.conv2d("conv1", input, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)
        print("=> conv1: " + str(x.get_shape()))

        x = layers.conv2d("conv2", x,
            self.get_depth(1), 4, 2,
            self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)
        print("=> conv2: " + str(x.get_shape()))

        res_cnt = 1
        for i in range(self.n_layer):
            for j in range(self.n_res):
                name = "res%d" % res_cnt
                x = layers.simple_residual_block(name, x, 3,
                    layers.LeakyReLU, bn_partial,
                    self.spectral_norm, update_collection, self.reuse)
                x = bn_partial(name + "/bn", x)
                x = layers.LeakyReLU(x)
                res_cnt += 1
                print("=> " + name + ": " + str(x.get_shape()))

            x = layers.conv2d("conv%d" % (i+3), x, self.get_depth(i+2), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial("conv%d/bn" % (i+3), x)
            x = layers.LeakyReLU(x)
            print("=> conv{}:\t".format(i+3) + str(x.get_shape()))
        
        x = tf.reduce_mean(x, [1, 2])
        print("=> gap: " + str(x.get_shape()))

        # do not use spectral norm in output
        self.disc_out = layers.linear("disc/fc", x, 1,
                        0, update_collection, self.reuse)
        self.cls_out = layers.linear("class/fc", x, self.n_attr,
                        0, update_collection, self.reuse)
        return self.disc_out, self.cls_out