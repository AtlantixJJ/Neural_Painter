import tensorflow as tf
from lib import ops, layers, utils
from model import basic
import numpy as np

class SimpleConvolutionGenerator(basic.SequentialNN):
    def __init__(self, map_size=4, map_depth=64, out_size=128, out_dim=3, spectral_norm=1, cbn_project=True, norm_mtd="cbn", **kwargs):
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
        self.n_layer = int(np.log2(out_size)) - int(np.log2(map_size))
        self.cbn_project = cbn_project
        self.norm_mtd = norm_mtd
        self.phase = "1"

        # use different output kernel size for different output size
        if self.out_size <= 64: self.ksize = 3
        elif self.out_size <= 128: self.ksize = 5
        else: self.ksize = 7
            
    def get_depth(self, i):
        """
        Given the layer index (start from 0 to self.n_layer), return the map depth
        """
        return min(2048, self.map_depth * (2 ** (self.n_layer - i)))

    def set_phase(self, phase):
        """
        When using batch norm in discrminator, the distribution of data in different phase is different
        """
        self.phase = phase

    def get_batchnorm(self):
        if self.norm_mtd == "cbn":
            # conditional bn: must use with conditional GAN
            return utils.partial(layers.conditional_batch_normalization, conditions=self.label, training=self.training, phase=self.phase, is_project=self.cbn_project, reuse=self.reuse)
        elif self.norm_mtd == "bn":
            return utils.partial(layers.default_batch_norm, training=self.training, phase=self.phase, reuse=self.reuse)
        elif self.norm_mtd == "default":
            def default_(name, x):
                return tf.layers.batch_normalization(x, training=self.training, name=name, reuse=self.reuse)
            return default_
        elif self.norm_mtd == "none":
            return lambda name, x: x

    def build_inference(self, input):
        self.input = input

        bn_partial = self.get_batchnorm()

        x = input
        x = self.check(x, "G/input")

        x = layers.linear("fc1", x, 
                (self.map_size ** 2) * self.get_depth(0),
                self.spectral_norm, self.reuse)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)
        print("=> fc1:\t" + str(x.get_shape()))
        x = self.check(x, "G/fc1")

        for i in range(self.n_layer):
            name = "deconv%d" % (i+1)
            x = layers.deconv2d(name, x, self.get_depth(i+1), 3, 2,
                self.spectral_norm, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ":\t" + str(x.get_shape()))
            x = self.check(x, "G/" + name)

        x = layers.conv2d("conv1", x, self.out_dim, self.ksize, 1, 0, None, self.reuse)
        print("=> conv1:\t" + str(x.get_shape()))
        x = self.check(x, "G/output")
        self.out = tf.nn.tanh(x)

        return self.out

class SimpleUpsampleGenerator(SimpleConvolutionGenerator):
    def __init__(self, **kwargs):
        """
        params:
        map_size:   the edge size of noise map
        map_depth:  smallest feature map depth (a unit)
        n_layer:   every one more convolution layer indicates a double in resolution
        """
        super(SimpleUpsampleGenerator, self).__init__(**kwargs)
    
    def build_inference(self, input):
        self.input = input

        bn_partial = self.get_batchnorm()

        x = input
        x = self.check(x, "G/input")

        x = layers.linear("fc1", x, 
                (self.map_size ** 2) * self.get_depth(0),
                self.spectral_norm, self.reuse)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.get_depth(0)])
        x = bn_partial('fc1/bn', x)
        x = tf.nn.relu(x)
        print("=> fc1:\t" + str(x.get_shape()))
        x = self.check(x, "G/fc1")

        for i in range(self.n_layer):
            name = "deconv%d" % (i+1)
            h, w, c = x.get_shape()[1:]
            x = tf.image.resize_nearest_neighbor(x, (h * 2, w * 2))
            x = layers.conv2d(name, x, self.get_depth(i+1), 3, 1,
                self.spectral_norm, self.reuse)
            x = bn_partial(name + "/bn", x)
            x = tf.nn.relu(x)
            print("=> " + name + ":\t" + str(x.get_shape()))
            x = self.check(x, "G/" + name)

        x = layers.conv2d("conv1", x, self.out_dim, self.ksize, 1, 0, self.reuse)
        print("=> conv1:\t" + str(x.get_shape()))
        x = self.check(x, "G/output")
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
    def __init__(self, n_attr=34, map_depth=64, map_size=4, input_size=128, spectral_norm=1, norm_mtd="none", **kwargs):
        super(SimpleConvolutionDiscriminator, self).__init__(**kwargs)

        self.spectral_norm = spectral_norm
        self.input_size = input_size
        self.map_size = map_size
        self.map_depth = map_depth
        self.n_layer = int(np.log2(input_size)) - int(np.log2(map_size))
        self.n_attr = n_attr
        self.label = None
        self.norm_mtd = norm_mtd
        self.phase = "1"

        # use different input kernel size for different input size
        if self.input_size <= 64: self.ksize = 3
        elif self.input_size <= 128: self.ksize = 5
        else: self.ksize = 7

    def set_phase(self, phase):
        """
        When using batch norm in discrminator, the distribution of data in different phase is different
        """
        self.phase = phase

    def set_label(self, label):
        self.label = label

    def get_depth(self, i):
        """
        Given the layer index (start from 0 to self.n_layer), return the map depth
        """
        return min(2048, self.map_depth * (2 ** i))

    def get_discriminator_batchnorm(self):
        """
        Define the bn function here, because bn is quite different.
        This should return a partial function with (name, x) as argument and other argument filled
        """

        def id_(name, x): return x

        def caffe_batch_norm_(name, x):
            return layers.caffe_batch_norm(name, x, phase=self.phase, training=self.training, reuse=self.reuse)

        def simple_batch_norm_(name, x):
            return layers.simple_batch_norm(name, x, phase=self.phase, training=self.training, reuse=self.reuse)

        def default_(name, x):
            return layers.default_batch_norm(name, x, phase=self.phase, reuse=self.reuse)

        def cbn_(name, x):
            return layers.conditional_batch_normalization(name, x, self.label, self.phase, is_project=True, reuse=self.reuse)

        if self.norm_mtd == "cbn": return cbn_
        elif self.norm_mtd == "bn": return default_
        elif self.norm_mtd == "none": return id_

    def build_inference(self, input):
        # usually discriminator do not use bn
        bn_partial = self.get_discriminator_batchnorm()

        x = tf.identity(input)
        x = self.check(x, "D/input/" + self.phase)
  
        x = layers.conv2d("conv1", x, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, self.reuse)
        x = bn_partial("bn1", x)
        x = layers.LeakyReLU(x)
        print("=> conv1:\t" + str(x.get_shape()))
        x = self.check(x, "D/conv1/" + self.phase)

        for i in range(self.n_layer):
            name = "conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.get_depth(i+1), 4, 2,
                self.spectral_norm, self.reuse)
            x = bn_partial("bn%d" % (i+2), x)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))
            x = self.check(x, "D/" + name + "/"  + self.phase)
        
        h = tf.reduce_mean(x, axis=[1, 2])
        print("=> disc/gap:\t" + str(h.get_shape()))

        h = self.check(h, "D/gap/" + self.phase)

        self.disc_out = layers.linear("disc/fc", h, 1,
                        0, self.reuse)
        self.cls_out = layers.linear("cls/fc", h, self.n_attr,
                        0, self.reuse)

        # class conditional info
        """
        if self.label is not None:
            dim = h.get_shape()[-1]
            emb_label = layers.linear("class/emd", self.label, dim,
                0, self.reuse)
            delta = tf.reduce_sum(h * emb_label, axis=[1], keepdims=True)
        else:
            delta = 0
        """

        if self.label is not None:
            return self.disc_out, self.cls_out
        else:
            return self.disc_out, 0

class SimpleDownsampleDiscriminator(SimpleConvolutionDiscriminator):
    def __init__(self, **kwargs):
        super(SimpleDownsampleDiscriminator, self).__init__(**kwargs)
    
    def build_inference(self, input, update_collection=None):
        # usually discriminator do not use bn
        bn_partial = self.get_discriminator_batchnorm()

        x = tf.identity(input)
        x = self.check(x, "D/input/" + self.phase)
  
        x = layers.conv2d("conv1", x, self.get_depth(0), self.ksize, 1,
                    self.spectral_norm, self.reuse)
        x = bn_partial("bn1", x)
        x = layers.LeakyReLU(x)
        print("=> conv1:\t" + str(x.get_shape()))
        x = self.check(x, "D/conv1/" + self.phase)

        for i in range(self.n_layer):
            name = "conv%d" % (i+2)
            x = layers.conv2d(name, x,
                self.get_depth(i+1), 3, 1,
                self.spectral_norm, self.reuse)
            x = tf.nn.avg_pool(x, 2, 2, "VALID")
            x = bn_partial("bn%d" % (i+2), x)
            x = layers.LeakyReLU(x)
            print("=> " + name + ":\t" + str(x.get_shape()))
            x = self.check(x, "D/" + name + "/"  + self.phase)
        
        h = tf.reduce_mean(x, axis=[1, 2])
        h = self.check(h, "D/gap/" + self.phase)
        print("=> disc/gap:\t" + str(h.get_shape()))

        self.disc_out = layers.linear("disc/fc", h, 1,
                        0, self.reuse)
        if self.label is not None:
            self.cls_out = layers.linear("cls/fc", h, self.n_attr,
                        0, self.reuse)

        # class conditional info
        """
        if self.label is not None:
            dim = h.get_shape()[-1]
            emb_label = layers.linear("class/emd", self.label, dim,
                0, update_collection, self.reuse)
            delta = tf.reduce_sum(h * emb_label, axis=[1], keepdims=True)
        else:
            delta = 0
        """

        # There is a self.disc_out = x. Effect not tested

        if self.label is not None:
            return self.disc_out, self.cls_out
        else:
            return self.disc_out, 0