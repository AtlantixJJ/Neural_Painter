import tensorflow as tf

from tensorflow.contrib import layers as L
import numpy as np
import math
from model.simple_generator import SimpleConvolutionGenerator, SimpleConvolutionDiscriminator

from lib import ops
from model import basic

class UNet(SimpleConvolutionGenerator):
    def __init__(self, batch_before=False, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.batch_before = batch_before

    def build_inference(self, input):
        x = input
        conv_cnt = 0
        def conv(x, ndim, ks, kt):
            conv_cnt += 1
            return L.conv2d(x, ndim, ks, kt,
                padding='SAME',
                activation_fn=None,
                reuse=self.reuse, scope="conv%d"%conv_cnt)

        e0 = tf.nn.relu(ops.get_norm(conv(x, 32, 3, 1), name+"/"+self.norm_mtd, self.training, self.reuse))
        e1 = tf.nn.relu(ops.get_norm(conv(e0, 64, 4, 2), ))
        e2 = tf.nn.relu(ops.get_norm(conv(e1, 64, 3, 1), ))
        del e1
        e3 = tf.nn.relu(ops.get_norm((conv(e2)))
        e4 = tf.nn.relu(ops.get_norm((conv(e3)))
        del e3
        e5 = tf.nn.relu(ops.get_norm((conv(e4)))
        e6 = tf.nn.relu(ops.get_norm((conv(e5)))
        del e5
        e7 = tf.nn.relu(ops.get_norm((conv(e6)))
        e8 = tf.nn.relu(ops.get_norm((conv(e7)))

        d8 = tf.nn.relu(ops.get_norm((self.dc8(F.concat([e7, e8]))))
        del e7, e8
        d7 = tf.nn.relu(ops.get_norm((self.dc7(d8)))
        del d8
        d6 = tf.nn.relu(ops.get_norm((self.dc6(F.concat([e6, d7]))))
        del d7, e6
        d5 = tf.nn.relu(ops.get_norm((self.dc5(d6)))
        del d6
        d4 = tf.nn.relu(ops.get_norm((self.dc4(F.concat([e4, d5]))))
        del d5, e4
        d3 = tf.nn.relu(ops.get_norm((self.dc3(d4)))
        del d4
        d2 = tf.nn.relu(ops.get_norm(2(self.dc2(F.concat([e2, d3]))))
        del d3, e2
        d1 = tf.nn.relu(ops.get_norm(1(self.dc1(d2)))
        del d2
        d0 = self.dc0(F.concat([e0, d1]))