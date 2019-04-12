import tensorflow as tf

from tensorflow.contrib import layers as L
import numpy as np
import math
from model.simple import SimpleConvolutionDiscriminator, SimpleConvolutionGenerator

from lib import ops, files, layers, utils
from model import basic

# not good
class ResidualGenerator(SimpleConvolutionGenerator):
    def __init__(self, **kwargs):
        super(ResidualGenerator, self).__init__(**kwargs)
    
    def build_inference(self, input, update_collection=None):
        # conditional bn: must use with conditional GAN
        #cbn_partial = utils.partial(layers.conditional_batch_normalization, conditions=input, phase=self.phase, update_collection=update_collection, is_project=self.cbn_project, reuse=self.reuse)
        bn_partial = utils.partial(layers.default_batch_norm, phase=self.phase, update_collection=update_collection, reuse=self.reuse)
        cbn_partial = bn_partial

        x = layers.linear("fc1", input, 
                (self.map_size ** 2) * self.get_depth(0),
                self.spectral_norm, update_collection, self.reuse)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)
        print("=> fc1:\t" + str(x.get_shape()))

        for i in range(self.n_layer + 1):
            #res1
            name = "res%d" % (i+1)
            x = layers.upsample_residual_block(name, x, self.get_depth(i),
                tf.nn.relu, cbn_partial,
                self.spectral_norm, update_collection, self.reuse)
            x = cbn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

            #res2
            """
            name = "res%d" % (2*i+1)
            x = layers.upsample_residual_block(name, x, self.get_depth(i),
                tf.nn.relu, cbn_partial,
                self.spectral_norm, update_collection, self.reuse)
            print("=> " + name + ":\t" + str(x.get_shape()))
            name = "res%d" % (2*i+2)
            x = layers.simple_residual_block(name, x, 3,
                tf.nn.relu, cbn_partial,
                self.spectral_norm, update_collection, self.reuse)
            print("=> " + name + ":\t" + str(x.get_shape()))
            """
            
        x = bn_partial("out/bn", x)
        x = tf.nn.relu(x)
        x = layers.conv2d("conv1", x, self.out_dim, self.ksize, 1, self.spectral_norm, update_collection, self.reuse)

        self.out = tf.nn.tanh(x)

        return self.out

# not good
class ResidualDiscriminator(SimpleConvolutionDiscriminator):
    def __init__(self, **kwargs):
        super(ResidualDiscriminator, self).__init__(**kwargs)
        
    def build_inference(self, input, update_collection=None):
        # usually discriminator do not use bn
        #bn_partial = utils.partial(layers.get_norm, training=self.training, reuse=self.reuse)
        def bn_partial(name, x): return x

        x = layers.conv2d("conv1", input, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, update_collection, self.reuse)
        x = bn_partial("bn1", x)
        x = layers.LeakyReLU(x)
        print("=> conv1:\t" + str(x.get_shape()))

        self.mid_layers = self.n_layer // 2 + 1

        for i in range(self.n_layer + 1):
            # res 1
            name = "res%d" % (i+1)
            x = layers.downsample_residual_block(name, x, self.get_depth(i),
                layers.LeakyReLU, bn_partial,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

            # res 2
            """
            name = "res%d" % (2*i+1)
            x = layers.downsample_residual_block(name, x, self.get_depth(i),
                layers.LeakyReLU, bn_partial,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

            name = "res%d" % (2*i+2)
            x = layers.simple_residual_block(name, x, 3,
                layers.LeakyReLU, bn_partial,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))
            """

        x = layers.LeakyReLU(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        print("=> disc:\t" + str(x.get_shape()))

        self.disc_out = layers.linear("disc/fc", x, 1,
                        self.spectral_norm, update_collection, self.reuse)

        self.cls_out = layers.linear("class/fc", x, self.n_attr, 
                    self.spectral_norm, update_collection, self.reuse)

        print("=> disc:\t" + str(self.disc_out.get_shape()))

        return self.disc_out, self.cls_out

class DeepGenerator(SimpleConvolutionGenerator):
    def __init__(self, n_res=2, **kwargs):
        super(DeepGenerator, self).__init__(**kwargs)
        self.n_res = n_res
        
    def build_inference(self, input, update_collection=None):
        # conditional bn: must use with conditional GAN
        cbn_partial = utils.partial(layers.conditional_batch_normalization, conditions=input, phase=self.phase, update_collection=update_collection, is_project=self.cbn_project, reuse=self.reuse)
        bn_partial = utils.partial(layers.default_batch_norm, phase=self.phase, update_collection=update_collection, reuse=self.reuse)

        x = layers.linear("fc1", input, 
                    (self.map_size ** 2) * self.map_depth,
                    self.spectral_norm, update_collection, self.reuse)
        x = tf.reshape(x,
            shape=[-1, self.map_size, self.map_size, self.map_depth])
        x = bn_partial("fc1/bn", x)
        x = tf.nn.relu(x)
        print("=> fc1: " + str(x.get_shape()))

        for i in range(self.n_layer // 2):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.map_depth // (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = cbn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ": " + str(x.get_shape()))
            
        base_x = tf.identity(x)

        res_cnt = 1
        for i in range(self.n_layer):
            x_id = tf.identity(x)
            for j in range(self.n_res):
                name = "Res%d" % res_cnt
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
        x = bn_partial("bridge/bn", x)
        x = tf.nn.relu(x)

        for i in range(self.n_enlarge // 2, self.n_enlarge):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.map_depth // (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ": " + str(x.get_shape()))

        x = layers.conv2d("conv%d" % (self.n_enlarge), x, self.out_dim, ksize, 1,
                    self.spectral_norm,
                    update_collection,
                    reuse=self.reuse)

        with tf.name_scope("gen_out") as nsc:
            self.out = tf.nn.tanh(x, name=nsc)

        return self.out    

class DeepDiscriminator(SimpleConvolutionDiscriminator):
    """
    n_layer == n_blocks
    """
    def __init__(self, n_res=2, n_block=5, **kwargs):
        super(DeepDiscriminator, self).__init__(**kwargs)

        self.n_res = n_res
        self.n_block = n_block

    def build_inference(self, input, update_collection=None):
        #bn_partial = utils.partial(layers.get_norm, training=self.training, reuse=self.reuse)
        if self.n_layer <= 5: ksize = 5
        elif self.n_layer <= 6: ksize = 7
        else: ksize = 9
        
        def bn_partial(name, x): return x

        x = layers.conv2d("main/conv1", input, self.map_depth, ksize, 1,
                    self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)
        print("=> main/conv1: " + str(x.get_shape()))

        res_cnt = 1
        self.mid_layer = self.n_layer // 2

        for i in range(self.mid_layer):
            name = "main/conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print("=> " + name + ": " + str(x.get_shape()))

        for i in range(self.n_block // 2):
            x_id = tf.identity(x)
            for j in range(self.n_res):
                name = "main/Res%d" % res_cnt
                x = layers.simple_residual_block(name, x, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                x = bn_partial(name + "/bn", x)
                x = layers.LeakyReLU(x)
                res_cnt += 1
                print("=> " + name + ": " + str(x.get_shape()))
            x = tf.add(x, x_id, "add")

        name = "main/conv%d" % (self.mid_layer+2)
        x = layers.conv2d(name, x,
            self.map_depth * (2 ** (self.mid_layer+1)), 4, 2,
            self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)
        print("=> " + name + ": " + str(x.get_shape()))

        class_branch = tf.identity(x)
        res_cnt1 = res_cnt
 
        for i in range(self.n_block // 2, self.n_block):
            for j in range(self.n_res):
                name = "disc/Res%d" % res_cnt
                x = layers.simple_residual_block(name, x, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                x = bn_partial(name + "/bn", x)
                x = layers.LeakyReLU(x)
                res_cnt += 1
                print("=> " + name + ": " + str(x.get_shape()))

        for i in range(self.mid_layer + 1, self.n_layer):
            name = "disc/conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print("=> " + name + ": " + str(x.get_shape()))
        
        # discrimination branch
        x = layers.learned_sum("disc/learned_sum", x, self.reuse)

        res_cnt = res_cnt1
        for i in range(self.n_block // 2, self.n_block):
            for j in range(self.n_res):
                name = "class/Res%d" % res_cnt
                class_branch = layers.simple_residual_block(name, class_branch, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                class_branch = bn_partial(name + "/bn", class_branch)
                class_branch = layers.LeakyReLU(class_branch)
                res_cnt += 1
                print("=> " + name + ": " + str(class_branch.get_shape()))

        for i in range(self.mid_layer + 1, self.n_layer):
            name = "class/conv%d" % (i+2)
            class_branch = layers.conv2d(name, class_branch,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            class_branch = layers.LeakyReLU(class_branch)
            print("=> " + name + ": " + str(class_branch.get_shape()))

        class_branch = layers.learned_sum("class/learned_sum", class_branch, self.reuse)

        # do not use spectral norm in output
        self.disc_out = layers.linear("fc_disc", x, 1,
                        0, update_collection, self.reuse)

        
        return self.disc_out, self.cls_out