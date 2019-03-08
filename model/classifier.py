import tensorflow as tf
from tensorflow.contrib import layers as L
from lib import ops
import numpy as np
from lib.tensorflowvgg.vgg16 import Vgg16

VGG_MEAN = [103.939, 116.779, 123.68]

class DilatedVGG16(Vgg16):
    def __init__(self, vgg16_npy_path):
        super(DilatedVGG16, self).__init__(vgg16_npy_path)

    def build(self, rgb):
        rgb_scaled = (rgb + 1) * 127.5

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.rate =  1

        self.conv1_1 = self.dilated_conv_layer(bgr, self.rate, "conv1_1")
        self.conv1_2 = self.dilated_conv_layer(self.conv1_1, self.rate, "conv1_2")
        #self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.rate *= 2

        self.conv2_1 = self.dilated_conv_layer(self.conv1_2, self.rate, "conv2_1")
        self.conv2_2 = self.dilated_conv_layer(self.conv2_1, self.rate, "conv2_2")
        #self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        self.rate *= 2

        self.conv3_1 = self.dilated_conv_layer(self.conv2_2, self.rate, "conv3_1")
        self.conv3_2 = self.dilated_conv_layer(self.conv3_1, self.rate, "conv3_2")
        self.conv3_3 = self.dilated_conv_layer(self.conv3_2, self.rate, "conv3_3")
        #self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        self.rate *= 2

        #self.conv4_1 = self.dilated_conv_layer(self.conv3_3, self.rate, "conv4_1")
        #self.conv4_2 = self.dilated_conv_layer(self.conv4_1, self.rate, "conv4_2")
        #self.conv4_3 = self.dilated_conv_layer(self.conv4_2, self.rate, "conv4_3")
        #self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        self.rate *= 2

        #self.conv5_1 = self.dilated_conv_layer(self.pool4, self.rate, "conv5_1")
        #self.conv5_2 = self.dilated_conv_layer(self.conv5_1, self.rate, "conv5_2")
        #self.conv5_3 = self.dilated_conv_layer(self.conv5_2, self.rate, "conv5_3")
        #self.conv5_4 = self.dilated_conv_layer(self.conv5_3, self.rate, "conv5_4")
        #self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        #print(self.pool5.get_shape())

        self.data_dict = None

class MyVGG16(Vgg16):
    def __init__(self, vgg16_npy_path):
        super(MyVGG16, self).__init__(vgg16_npy_path)

    def build(self, rgb):
        rgb_scaled = (rgb + 1) * 127.5

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")

        self.data_dict = None
