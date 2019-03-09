import tensorflow as tf
from lib import ops, layers, utils
from model import basic
import numpy as np

class SimpleConvolutionGenerator(basic.SequentialNN):
    def __init__(self, map_size=4, map_depth=1024, n_layer=5, out_dim=3, spectral_norm=True, **kwargs):
        """
        params:
        map_size:   the edge size of noise map
        map_depth:  initial depth of first feature map
        n_layer:   every one more convolution layer indicates a double in resolution
        """
        super(SimpleConvolutionGenerator, self).__init__(**kwargs)
        
        self.spectral_norm = spectral_norm
        self.out_dim = out_dim
        self.map_size = map_size
        self.map_depth = map_depth
        self.n_layer = n_layer
        self.cbn_project = False

    def build_inference(self, input, update_collection="no_ops"):
        if self.n_layer <= 5: ksize = 5
        elif self.n_layer <= 6: ksize = 7
        else: ksize = 9

        bn_partial = utils.partial(layers.get_norm, method=self.norm_mtd, training=self.training, reuse=self.reuse)

        # ReLU is default in this function
        x = layers.linear("fc1", input, 
                (self.map_size ** 2) * self.map_depth,
                self.spectral_norm, update_collection, self.reuse)

        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.map_depth])
        #x = layers.conditional_batch_normalization("cbn", x, input, self.cbn_project, self.reuse)
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)

        for i in range(1, self.n_layer + 1, 1):
            name = "deconv%d" % i
            x = layers.deconv2d(name, x, int(self.map_depth // (2 ** i)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = bn_partial(name + "/bn", x)
            #x = layers.conditional_batch_normalization(name + "cbn", x, input, self.cbn_project, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())

        x = layers.conv2d("conv1", x, self.out_dim, ksize, 1, self.spectral_norm, update_collection, self.reuse)

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
    def __init__(self, n_layer=5, n_attr=34, map_depth=64, spectral_norm=True, **kwargs):
        super(SimpleConvolutionDiscriminator, self).__init__(**kwargs)

        self.spectral_norm = spectral_norm
        self.map_depth = map_depth
        self.n_layer = n_layer
        self.n_attr = n_attr

    def build_inference(self, input, update_collection="no_ops"):
        #bn_partial = utils.partial(layers.get_norm, training=self.training, reuse=self.reuse)
        if self.n_layer <= 5: ksize = 5
        elif self.n_layer <= 6: ksize = 7
        else: ksize = 9

        x = layers.conv2d("conv1", input, self.map_depth, ksize, 1,
                    self.spectral_norm, update_collection, self.reuse)
        x = layers.LeakyReLU(x)

        self.mid_layers = self.n_layer // 2

        for i in range(self.mid_layers):
            x = layers.conv2d("main_conv%d" % (i+2), x,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print(x.get_shape())

        class_branch = tf.identity(x)

        for i in range(self.mid_layers, self.n_layer):
            x = layers.conv2d("disc_conv%d" % (i+2), x,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            x = layers.LeakyReLU(x)
            print(x.get_shape())
        
        # discrimination branch
        x = layers.learned_sum("ls_disc", x, self.reuse)

        for i in range(self.mid_layers, self.n_layer):
            class_branch = layers.conv2d("class_conv%d" % (i+2), class_branch,
                self.map_depth * (2 ** (i+1)), 4, 2,
                self.spectral_norm, update_collection, self.reuse)
            class_branch = layers.LeakyReLU(class_branch)
            print(x.get_shape())

        class_branch = layers.learned_sum("ls_class", class_branch, self.reuse)

        self.disc_out = layers.linear("fc_disc", x, 1,
                        self.spectral_norm, update_collection, self.reuse)

        self.cls_out = layers.linear("fc_cls", class_branch, self.n_attr, 
                    self.spectral_norm, update_collection, self.reuse)
        
        return self.disc_out, self.cls_out