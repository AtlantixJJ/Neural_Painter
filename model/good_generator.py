import tensorflow as tf

from tensorflow.contrib import layers as L
import numpy as np
import math
from model.simple_generator import SimpleConvolutionDiscriminator, SimpleConvolutionGenerator

from lib import ops, files
from model import basic

class GoodGenerator(SimpleConvolutionGenerator):
    """
    GoodGenerator. 8x upsample.
    Args:
    n_sample:The input noise's dimension[128]
    map_size:The target's width(height)/16[4] 
    gf_dim:output dimension of gen filters in last deconv layer. [64]
    out_dim:Dimension of image color. For grayscale input, set to 1. [3]
    """
    def __init__(self, batch_before=False, **kwargs):
        super(GoodGenerator, self).__init__(**kwargs)

        self.batch_before = batch_before

        self.has_bridge = True
        
    def build_inference(self, input, scope=None):
        x = input
        output = base_output = 0
        if self.batch_before:
            output = ops.linear("fc1", x, 
                        (self.map_size ** 2) * self.map_depth,
                        activation_fn=tf.nn.relu,
                        normalizer_mode=self.norm_mtd,
                        training=self.training,
                        reuse=self.reuse)

            base_output = tf.reshape(output,
                shape=[-1, self.map_size, self.map_size, self.map_depth])
        else:
            output = ops.linear("fc1", x, 
                        (self.map_size ** 2) * self.map_depth,
                        activation_fn=None,
                        normalizer_mode=None,
                        training=self.training,
                        reuse=self.reuse)

            output = tf.reshape(output,
                shape=[-1, self.map_size, self.map_size, self.map_depth])
            
            output = ops.get_norm(output,
                name="fc1/"+self.norm_mtd, training=self.training, reuse=self.reuse)
            output = tf.nn.relu(output)

            base_output = tf.identity(output)

        output = tf.identity(base_output, name="bridge_base")

        #output = ops.get_norm(output, "fc1/"+self.norm_mtd, self.training, self.reuse)
        

        for i in range(self.n_layers):
            output = ops.residual_block(name="Res%d" % (i+1),
                input=output,
                output_dim=self.map_depth,
                filter_size=3,
                activation_fn=tf.nn.relu,
                normalizer_mode=self.norm_mtd,
                training=self.training,
                resample=None,
                reuse=self.reuse)
            #output = tf.nn.relu(output)
        if self.has_bridge:
            with tf.name_scope("bridge"):
                output = ops.get_norm(output,
                    name=self.norm_mtd, training=self.training, reuse=self.reuse)
                output = tf.nn.relu(output)
                output = tf.add(output, base_output, name="bridge_join")
        else:
            output = ops.get_norm(output,
                    name=self.norm_mtd, training=self.training, reuse=self.reuse)

        for i in range(self.n_enlarge-1):
            output = ops.subpixel_conv("deconv%d" % (i+1), output, self.map_depth, 
                        activation_fn=tf.nn.relu,
                        normalizer_mode=self.norm_mtd,
                        training=self.training,
                        reuse=self.reuse)

        output = ops.conv2d("deconv%d" % (self.n_enlarge), output, self.out_dim, 9, 1,
                    activation_fn=None,
                    normalizer_mode=None,
                    training=self.training,
                    reuse=self.reuse)

        with tf.name_scope("gen_out") as nsc:
            if self.out_fn == 'tanh':
                self.out = tf.nn.tanh(output, name=nsc)
            elif self.out_fn == 'relu':
                self.out = tf.nn.relu(output, name=nsc)
        
        return self.out    

class GoodDiscriminator(SimpleConvolutionDiscriminator):
    """
    n_layers == n_blocks
    """
    def __init__(self, n_res=2, **kwargs):
        super(GoodDiscriminator, self).__init__(**kwargs)

        self.n_res = n_res
        self.n_blocks = self.n_layers

    def build_inference(self, input, scope=None):
        x = input

        output = ops.conv2d("conv1", x, self.map_depth, 5, 1,
                        activation_fn=ops.LeakyReLU,
                        normalizer_mode=self.norm_mtd,
                        training=self.training,
                        reuse=self.reuse)
        print(output.get_shape())
        # 2x
        res_cnt = 0

        for i in range(self.n_blocks):
            for j in range(self.n_res):
                output = ops.residual_block(name="Res%d" % res_cnt,
                    input=output,
                    filter_size=3,
                    activation_fn=ops.LeakyReLU,
                    normalizer_mode=self.norm_mtd,
                    resample=None,
                    training=self.training,
                    reuse=self.reuse)
                output = ops.LeakyReLU(output)
                res_cnt = res_cnt + 1

            output = ops.conv2d("conv%d" % (i+2), output, self.map_depth * (2 ** i), 4, 2,
                        activation_fn=ops.LeakyReLU,
                        normalizer_mode=self.norm_mtd,
                        training=self.training,
                        reuse=self.reuse)
            print(output.get_shape())
        
        #output = tf.reshape(output, shape=[-1, np.prod(output.get_shape().as_list()[1:])])
        output = tf.reduce_mean(output, axis=[1, 2])
        self.build_tail(output)

        return self.disc_out, self.cls_out
