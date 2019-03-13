import tensorflow as tf
from lib import ops, layers, utils
from model import basic
import numpy as np

class SimpleConvolutionGenerator(basic.SequentialNN):
    def __init__(self, map_size=4, map_depth=64, out_size=128, out_dim=3, spectral_norm=True, cbn_project=True, **kwargs):
        """
        params:
        map_size:   the edge size of noise map
        map_depth:  smallest feature map depth (a unit)
        n_layer:   every one more convolution layer indicates a double in resolution
        """
        super(SimpleConvolutionGenerator, self).__init__(**kwargs)
        
        self.spectral_norm = spectral_norm
        self.out_dim = out_dim
        self.out_size = out_size
        self.map_size = map_size
        self.map_depth = map_depth
        self.n_layer = int(np.log2(out_size)) - int(np.log2(map_size)) - 1
        self.cbn_project = cbn_project

        # use different output kernel size for different output size
        if self.out_size <= 32: self.ksize = 3
        elif self.out_size <= 64: self.ksize = 5
        elif self.out_size <= 128: self.ksize = 7
        else: self.ksize = 9
            
    def get_depth(self, i):
        """
        Given the layer index (start from 0 to self.n_layer), return the map depth
        """
        return self.map_depth * (2 ** (self.n_layer - i))

    def build_inference(self, input, update_collection="no_ops"):
        # normal bn
        #bn_partial = utils.partial(layers.get_norm, method=self.norm_mtd, training=self.training, reuse=self.reuse)
        # conditional bn: must use with conditional GAN
        bn_partial = utils.partial(layers.conditional_batch_normalization, conditions=input, training=self.training, is_project=self.cbn_project, reuse=self.reuse)
        # partial function: fill in some argument in advance, and only (name, input) is needed at call time

        x = layers.linear("fc1", input, 
                (self.map_size ** 2) * self.get_depth(0),
                self.spectral_norm, update_collection, self.reuse)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)
        print("=> fc1:\t" + str(x.get_shape()))

        for i in range(self.n_layer + 1):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.get_depth(i), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

        x = layers.conv2d("conv1", x, self.out_dim, self.ksize, 1, self.spectral_norm, update_collection, self.reuse)
        print("=> conv1:\t" + str(x.get_shape()))
        self.out = tf.nn.tanh(x)

        return self.out

### [WARNING]: This is not adapted to new API
class MaskConvolutionGenerator(basic.SequentialNN):
    def __init__(self, mask_num=4, map_size=4, map_depth=1024, n_layer=5, out_dim=3, **kwargs):
        """
        params:
        map_size:   the edge size of noise map
        map_depth:  initial depth of first feature map
        n_layer:   every one more convolution layer indicates a double in resolution
        """
        super(MaskConvolutionGenerator, self).__init__(**kwargs)

        self.mask_num = mask_num
        self.out_dim = out_dim
        self.map_size = map_size
        self.map_depth = map_depth
        self.n_layer = n_layer
        self.cbn_project = False

    def build_inference(self, input, update_collection="no_ops"):
        """
        update_collection is some assign ops that need to be done
        """
        # ReLU is default in this function
        x = ops.spectral_linear("fc1", input, 
                (self.map_size ** 2) * self.map_depth,
                reuse=self.reuse, update_collection=update_collection)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.map_depth])
        x = ops.conditional_batch_normalization("cbn", x, input, self.cbn_project, self.reuse)
        x = tf.nn.relu(x)

        self.mid_layers = (self.n_layer + 1) // 2

        for i in range(1, self.mid_layers):
            name = "main_deconv%d" % i
            x = ops.spectral_deconv2d(name, x, int(self.map_depth // (2 ** i)), 4, 2,
                reuse=self.reuse, update_collection=update_collection)
            x = ops.conditional_batch_normalization(name + "/cbn", x, input, self.cbn_project, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())

        mask_x = tf.identity(x)

        for i in range(self.mid_layers, self.n_layer + 1):
            name = "draw_deconv%d" % i
            x = ops.spectral_deconv2d(name, x, int(self.map_depth // (2 ** i)), 4, 2,
                reuse=self.reuse, update_collection=update_collection)
            x = ops.conditional_batch_normalization(name + "/cbn", x, input, self.cbn_project, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())

        for i in range(self.mid_layers, self.n_layer + 1):
            name = "mask_deconv%d" % i
            mask_x = ops.spectral_deconv2d(name, mask_x, int(self.map_depth // (2 ** i)), 4, 2,
                reuse=self.reuse, update_collection=update_collection)
            mask_x = ops.conditional_batch_normalization(name + "/cbn", mask_x, input, self.cbn_project, self.reuse)
            mask_x = tf.nn.relu(mask_x)
            print(x.get_shape())

        size, depth = x.get_shape().as_list()[-2:]

        mask_logits = ops.spectral_conv2d("mask", mask_x, self.mask_num, 3, 1,
            reuse=self.reuse, update_collection=update_collection)
        self.mask_logits = mask_logits
        self.overlapped_mask = tf.nn.softmax(mask_logits, axis=3)
        
        with tf.variable_scope("mask_stroke", reuse=self.reuse):
            shape = [self.mask_num, 3, 3, depth, self.out_dim]
            mask_conv_filters = tf.get_variable("mask_conv_kernels",
                shape=shape,
                initializer=tf.orthogonal_initializer)

        self.masks = []
        self.outs = []

        for i in range(self.mask_num):
            mask_e = self.overlapped_mask[:, :, :, i]
            with tf.variable_scope("mask_norm%d" % i, reuse=self.reuse):
                w = ops.spectral_normed_weight(mask_conv_filters[i], update_collection=update_collection)
            out_e = tf.nn.conv2d(x, w,
                [1, 1, 1, 1],
                "SAME")
            self.outs.append(out_e * tf.expand_dims(mask_e, -1))
            self.masks.append(mask_e)

        self.out = tf.nn.tanh(sum(self.outs))

        return self.out

class SimpleConvolutionDiscriminator(basic.SequentialNN):
    def __init__(self, n_attr=34, map_depth=64, map_size=4, input_size=128, spectral_norm=True, **kwargs):
        super(SimpleConvolutionDiscriminator, self).__init__(**kwargs)

        self.spectral_norm = spectral_norm
        self.input_size = input_size
        self.map_size = map_size
        self.map_depth = map_depth
        self.n_layer = int(np.log2(input_size)) - int(np.log2(map_size)) - 1
        self.n_attr = n_attr

        # use different input kernel size for different input size
        if self.input_size <= 32: self.ksize = 3
        elif self.input_size <= 64: self.ksize = 5
        elif self.input_size <= 128: self.ksize = 7
        else: self.ksize = 9

    def get_depth(self, i):
        """
        Given the layer index (start from 0 to self.n_layer), return the map depth
        """
        return self.map_depth * (2 ** i)

    def build_inference(self, input, update_collection=None):
        # usually discriminator do not use bn
        #bn_partial = utils.partial(layers.get_norm, training=self.training, reuse=self.reuse)

        x = layers.conv2d("main/conv1", input, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)
        print("=> main/conv1:\t" + str(x.get_shape()))

        self.mid_layers = self.n_layer // 2 + 1

        for i in range(self.mid_layers):
            name = "main/conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.get_depth(i), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

        class_branch = tf.identity(x)

        for i in range(self.mid_layers, self.n_layer + 1):
            name = "disc/conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.get_depth(i), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))

        for i in range(self.mid_layers, self.n_layer + 1):
            name = "class/conv%d" % (i+2)
            class_branch = layers.conv2d(name, class_branch,
                self.get_depth(i), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            class_branch = layers.LeakyReLU(class_branch)
            print("=> " + name + ":\t" + str(class_branch.get_shape()))

        class_branch = tf.reduce_mean(class_branch, axis=[1, 2])
        print("=> class:\t" + str(class_branch.get_shape()))
        x = tf.reduce_mean(x, axis=[1, 2])
        print("=> disc:\t" + str(x.get_shape()))

        self.disc_out = layers.linear("disc/fc", x, 1,
                        self.spectral_norm, update_collection, self.reuse)

        self.cls_out = layers.linear("class/fc", class_branch, self.n_attr, 
                    self.spectral_norm, update_collection, self.reuse)

        print("=> class:\t" + str(self.cls_out.get_shape()))
        print("=> disc:\t" + str(self.disc_out.get_shape()))

        return self.disc_out, self.cls_out