import tensorflow as tf

from tensorflow.contrib import layers as L
import numpy as np
import math
from model.simple import SimpleConvolutionDiscriminator, SimpleConvolutionGenerator

from lib import ops, files, layers, utils
from model import basic

class DeepGenerator(SimpleConvolutionGenerator):
    """
    DeepGenerator. 8x upsample.
    Args:
    n_sample:The input noise's dimension[128]
    map_size:The target's width(height)/16[4] 
    gf_dim:output dimension of gen filters in last deconv layer. [64]
    out_dim:Dimension of image color. For grayscale input, set to 1. [3]
    """
    def __init__(self, n_enlarge, **kwargs):
        super(DeepGenerator, self).__init__(**kwargs)
        self.n_enlarge = n_enlarge
        
    def build_inference(self, input, update_collection=None):
        if self.n_enlarge <= 3: ksize = 5
        elif self.n_enlarge <= 4: ksize = 7
        else: ksize = 9

        x = input

        bn_partial = utils.partial(layers.get_norm, method=self.norm_mtd, training=self.training, reuse=self.reuse)
        #cbn_partial = utils.partial(layers.conditional_batch_normalization, conditions=input, training=self.training, is_projection=self.cbn_project, reuse=self.reuse)

        output = layers.linear("fc1", x, 
                    (self.map_size ** 2) * self.map_depth,
                    self.spectral_norm, update_collection, self.reuse)
        output = tf.reshape(output,
            shape=[-1, self.map_size, self.map_size, self.map_depth])
        
        output = bn_partial("fc1/bn", output)
        output = tf.nn.relu(output)
        print(output.get_shape())

        output = layers.deconv2d("conv1", output, self.map_depth // 2, 4, 2,
            self.spectral_norm, update_collection, self.reuse)
        output = bn_partial("conv1/bn", output)
        output = tf.nn.relu(output)
        print(output.get_shape())

        base_output = tf.identity(output)

        output = tf.identity(base_output, name="bridge_base")

        for i in range(self.n_layer):
            name = "Res%d" % (i+1)
            output = layers.simple_residual_block(name, output, 3,
                tf.nn.relu,
                bn_partial,
                self.spectral_norm,
                update_collection,
                self.reuse)
            output = bn_partial(name + "/bn", output)
            output = tf.nn.relu(output)
            
        output = bn_partial("bridge/bn", output)
        output = tf.nn.relu(output)
        output = tf.add(output, base_output, name="bridge_join")

        for i in range(self.n_enlarge-1):
            name = "deconv%d" % (i+1)
            output = layers.deconv2d(name, output, self.map_depth // (2 ** (i+2)), 4, 2,
                        self.spectral_norm,
                        update_collection,
                        self.reuse)
            output = bn_partial(name + "/bn", output)
            output = tf.nn.relu(output)
            print(output.get_shape())

        output = layers.conv2d("deconv%d" % (self.n_enlarge), output, self.out_dim, ksize, 1,
                    self.spectral_norm,
                    update_collection,
                    reuse=self.reuse)

        with tf.name_scope("gen_out") as nsc:
            self.out = tf.nn.tanh(output, name=nsc)

        return self.out    

class DeepDiscriminator(SimpleConvolutionDiscriminator):
    """
    n_layer == n_blocks
    """
    def __init__(self, n_res=2, **kwargs):
        super(DeepDiscriminator, self).__init__(**kwargs)

        self.n_res = n_res
        self.n_blocks = self.n_layer // self.n_res

    def build_inference(self, input, update_collection=None):
        if self.n_layer <= 6: ksize = 5
        elif self.n_layer <= 8: ksize = 7
        else: ksize = 9
        
        def bn_partial(name, x): return x
        #bn_partial = utils.partial(layers.get_norm, training=self.training, reuse=self.reuse)

        output = layers.conv2d("main/conv1", input, self.map_depth, ksize, 1,
                        self.spectral_norm,
                        update_collection,
                        reuse=self.reuse)
        #output = bn_partial("main/conv1/" + self.norm_mtd, output)
        output = layers.LeakyReLU(output)
        print(output.get_shape())

        # 2x
        res_cnt = 0

        for i in range(self.n_blocks // 2):
            for j in range(self.n_res):
                name = "main/Res%d/" % res_cnt
                output = layers.simple_residual_block(name, output, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                output = bn_partial(name + self.norm_mtd, output)
                output = layers.LeakyReLU(output)
                res_cnt = res_cnt + 1

            output = layers.conv2d("main/conv%d" % (i+2), output, self.map_depth * (2 ** (i+1)), 4, 2,
                        self.spectral_norm,
                        update_collection,
                        self.reuse)
            output = bn_partial("main/conv%d/%s" % (i+2, self.norm_mtd), output)
            output = layers.LeakyReLU(output)
            print(output.get_shape())
        
        class_branch = tf.identity(output, "class/start")
        res_cnt1 = res_cnt

        for i in range(self.n_blocks // 2, self.n_blocks):
            for j in range(self.n_res):
                name = "main/Res%d/" % res_cnt
                output = layers.simple_residual_block(name, output, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                output = bn_partial(name + self.norm_mtd, output)
                output = layers.LeakyReLU(output)
                res_cnt = res_cnt + 1

            output = layers.conv2d("main/conv%d" % (i+2), output, self.map_depth * (2 ** (i+1)), 4, 2,
                        self.spectral_norm,
                        update_collection,
                        self.reuse)
            output = bn_partial("main/conv%d/%s" % (i+2, self.norm_mtd), output)
            output = layers.LeakyReLU(output)
            print(output.get_shape())

        res_cnt = res_cnt1
        for i in range(self.n_blocks // 2, self.n_blocks):
            for j in range(self.n_res):
                name = "class/Res%d/" % res_cnt
                class_branch = layers.simple_residual_block(name, class_branch, 3,
                    layers.LeakyReLU,
                    bn_partial,
                    self.spectral_norm,
                    update_collection,
                    self.reuse)
                class_branch = bn_partial(name + self.norm_mtd, class_branch)
                class_branch = layers.LeakyReLU(class_branch)
                res_cnt = res_cnt + 1

            class_branch = layers.conv2d("class/conv%d" % (i+2), class_branch, self.map_depth * (2 ** (i+1)), 4, 2,
                        self.spectral_norm,
                        update_collection,
                        self.reuse)
            class_branch = bn_partial("class/conv%d/%s" % (i+2, self.norm_mtd), class_branch)
            class_branch = layers.LeakyReLU(class_branch)
            print(class_branch.get_shape())

        output = layers.learned_sum("main/learned_sum", output, self.reuse)
        class_branch = layers.learned_sum("class/learned_sum", class_branch, self.reuse)
        #output = tf.reshape(output, shape=[-1, np.prod(output.get_shape().as_list()[1:])])

        self.disc_out = layers.linear("fc_disc", output, 1,
                        self.spectral_norm, update_collection, self.reuse)

        self.cls_out = layers.linear("fc_cls", class_branch, self.n_attr, 
                    self.spectral_norm, update_collection, self.reuse)

        return self.disc_out, self.cls_out
