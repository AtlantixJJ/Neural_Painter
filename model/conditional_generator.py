"""
Deprecated
"""
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib import layers as L
from lib import ops
import numpy as np
from model import basic, simple_generator


class SimpleConditionalGenerator(basic.SequentialNN):
    """Shallow layer conditional generator

        Shallow layer conditional generator
    """
    def __init__(self, **kwargs):
        super(SimpleConditionalGenerator, self).__init__(**kwargs)

    def build_inference(self, input):
        """
        input is a list : [z, c]
        """
        x, label = input
        x = tf.concat([x, label], axis=1)
        with tf.variable_scope("fc1"):
            # ReLU is default in this function
            x = L.fully_connected(x, self.map_depth * self.map_size ** 2, normalizer_fn=L.batch_norm)
        x = tf.reshape(x, [-1, self.map_size, self.map_size, self.map_depth])

        for i in range(1, self.n_layer + 1, 1):
            with tf.variable_scope("deconv%d" % i):
                x = L.conv2d_transpose(x, int(self.map_depth / (2 ** i)), 3, 2, normalizer_fn=L.batch_norm,
                    weights_regularizer=L.l2_regularizer(self.lambda_l2))

        with tf.variable_scope("deconv%d" % (self.n_layer + 1)):
            x = L.conv2d_transpose(x, self.out_dim, 1, 1, activation_fn=None,
                weights_regularizer=L.l2_regularizer(self.lambda_l2))

        if self.out_fn == 'tanh':
            self.out = tf.nn.tanh(x)
        elif self.out_fn == 'relu':
            self.out = tf.nn.relu(x)
        # crop to 28x28
        # self.out = x[:, 2:-2, 2:-2, :]
        return self.out

    def build_train(self, input):
        with tf.variable_scope(self.name):
            self.build_inference(input)
        self.vars = [v for v in tf.trainable_variables() if self.name in v.name]


    def build_train_op(self, lr, cost):
        if cost != None:
            gen_cost, cls_cost = cost
            reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            cost = gen_cost + cls_cost + reg_losses
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.,
                beta2=0.9).minimize(cost, var_list=self.vars, colocate_gradients_with_ops=True)

class SimpleConditionalDiscriminator(basic.SequentialNN):
    """Shallow layer conditional discriminator

        Shallow layer conditional discriminator.
    """
    def __init__(self, **kwargs):
        super(SimpleConditionalDiscriminator, self).__init__(**kwargs)
        self.c_len = c_len
        self.reuse = False

    def build_inference(self, input):
        x = input
        for i in range(1, self.n_layer+1, 1):
            with tf.variable_scope("conv%d" % (self.n_layer - i), reuse=self.reuse):
                x = L.conv2d(x, self.map_depth / (2 ** (self.n_layer - i) ), 3, 2,
                    normalizer_fn=L.batch_norm,
                    weights_regularizer=L.l2_regularizer(self.lambda_l2),
                    activation_fn=ops.LeakyReLU)

        # now x is 4x4
        x = tf.reshape(x, [-1, self.map_size * self.map_size * self.map_depth])

        with tf.variable_scope("fc_disc%d" % (self.n_layer + 1), reuse=self.reuse):
            if self.is_wgan:
                self.disc_out = L.fully_connected(x, 1, activation_fn=None, weights_regularizer=L.l2_regularizer(self.lambda_l2))
            else:
                self.disc_out = L.fully_connected(x, 1, activation_fn=tf.nn.sigmoid, weights_regularizer=L.l2_regularizer(self.lambda_l2))
        
        # ADD for classification
        with tf.variable_scope("fc_cls%d" % (self.n_layer + 1), reuse=self.reuse):
            self.disc_cls = L.fully_connected(x, self.c_len, activation_fn=tf.nn.sigmoid, weights_regularizer=L.l2_regularizer(self.lambda_l2))
        return self.disc_out, self.disc_cls

    def build_train(self, input):
        # ADD label
        fake_data, real_data, label = input
        with tf.variable_scope(self.name):
            self.disc_fake, self.disc_fake_cls = self.build_inference(fake_data)
            self.reuse = True
            self.disc_real, self.disc_real_cls = self.build_inference(real_data)
        self.vars = [v for v in tf.trainable_variables() if self.name in v.name]

        self.tot_loss = 0
        
        if self.is_wgan:
            self.gen_cost = -tf.reduce_mean(self.disc_fake)
            self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1],
                minval=0.,
                maxval=1.
            )
            differences = real_data - fake_data
            interpolates = fake_data + alpha * differences#tf.multiply(alpha, differences)
            with tf.variable_scope(self.name):
                self.disc_interp = self.build_inference(interpolates)
            gradients = tf.gradients(self.disc_interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = self.lambda_gp * tf.reduce_mean((slopes-1.)**2)
            self.tot_loss += self.gradient_penalty

        else:
            # normal disc
            self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_fake,
                    labels=tf.ones_like(self.disc_fake)))
        
            self.disc_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_fake,
                    labels=tf.zeros_like(self.disc_fake)))

            self.disc_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_real,
                    labels=tf.ones_like(self.disc_real)))
            
            self.disc_cost = self.disc_cost_fake + self.disc_cost_real
        
        self.cls_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.disc_fake_cls,
                labels=label))
        self.cls_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.disc_real_cls,
                labels=label))

        reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.tot_loss += self.disc_cost + reg_losses + self.cls_cost
        

        with tf.name_scope(self.name + "_Loss"):
            l2_sum = tf.summary.scalar("L2 regularize", reg_losses)
            cls_sum = tf.summary.scalar("classification", self.cls_cost)
            if self.is_wgan:
                gp_sum = tf.summary.scalar("gradient penalty", self.gradient_penalty)
                disc_cost_sum = tf.summary.scalar("discriminator loss", self.disc_cost)
            else:
                disc_cost_real_sum = tf.summary.scalar("discriminator loss real", self.disc_cost_real)
                disc_cost_fake_sum = tf.summary.scalar("discriminator loss fake", self.disc_cost_fake)
            gen_cost_sum = tf.summary.scalar("generator loss", self.gen_cost)
            self.sum_op = tf.summary.merge_all()
            # grid summary 4x4
            show_img = ops.get_grid_image_summary(fake_data, 4)
            gen_image_sum = tf.summary.image("generated", show_img + 1)
            self.sum_interval_op = tf.summary.merge([gen_image_sum])
            

    def build_train_op(self, lr=1e-4, cost=None):
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.,
            beta2=0.9
        ).minimize(self.tot_loss, var_list=self.vars, colocate_gradients_with_ops=True)
class ImageConditionalDiscriminator(basic.SequentialNN):
    """Shallow layer DCGAN discriminator

        Shallow layer DCGAN discriminator.
    """
    def __init__(self, **kwargs):
        super(ImageConditionalDiscriminator, self).__init__(**kwargs)

        self.reuse = False

    def build_inference(self, input):
        if self.is_cgan:
            x, self.label = input
        else:
            x = input

        ndf = 32
        ksize = 4
        n_layer = 3
        depth = [ndf * 2, ndf * 4, ndf * 8]
        self.norm_mtd = ""

        print("Discriminator shape log:")

        x = L.conv2d(x, ndf, ksize, 2, padding='SAME', scope='conv1', reuse=self.reuse, activation_fn=None)
        x = ops.get_norm(x, "conv1/" + self.norm_mtd, self.training, self.reuse)
        x = ops.LeakyReLU(x)
        print(x.get_shape())

        for i in range(n_layer):
            name = "conv%d" % (i + 2)
            x = L.conv2d(x, depth[i], ksize, 2, padding='SAME', scope=name, reuse=self.reuse, activation_fn=None)
            x = ops.get_norm(x, name + "/" + self.norm_mtd, self.training, self.reuse)
            x = ops.LeakyReLU(x)

            #x = tf.nn.dropout(x, self.keep_prob)
            print(x.get_shape())

        x = L.conv2d(x, depth[-1], 3, 1, padding='VALID', scope="conv_trunk", reuse=self.reuse, activation_fn=None)
        x = ops.get_norm(x, name + "/" + self.norm_mtd, self.training, self.reuse)
        x = ops.LeakyReLU(x)
        print(x.get_shape())

        self.build_tail(x)

        return self.disc_out, self.cls_out

    def build_tail(self, x):
        #self.disc_out = tf.reduce_mean(
        #    L.conv2d(x, 1, (3, 3), 1, padding='VALID', scope="conv_disc", reuse=self.reuse),
        #    axis=[1,2,3])
        self.disc_out = L.conv2d(x, 1, 3, 1, padding='VALID', scope="conv_disc", reuse=self.reuse, activation_fn=None)
        print("Discriminator output shape:")
        print(self.disc_out.get_shape())
        ksize = x.get_shape().as_list()
        ksize[0] = ksize[3] = 1
        x = tf.nn.avg_pool(x, ksize, [1, 1, 1, 1], "VALID")
        print(x.get_shape())
        x = tf.reshape(x, [-1, x.get_shape()[-1]])
        self.cls_out = L.fully_connected(x, self.c_len, scope="fc_cls", reuse=self.reuse, activation_fn=None)
        print("Discriminator classification shape:")
        print(self.cls_out.get_shape())


class ImageConditionalDeepDiscriminator(ImageConditionalDiscriminator):
    """Shallow layer DCGAN discriminator

        Shallow layer DCGAN discriminator.
    """
    def __init__(self, **kwargs):
        super(ImageConditionalDeepDiscriminator, self).__init__(**kwargs)

        self.reuse = False

    def build_inference(self, x):
        ndf = 64
        ksize = 4
        layer_depth = [ndf * 4, ndf * 8, ndf * 16, ndf * 4, ndf * 8]
        self.norm_mtd = "inst"

        x = L.conv2d(x, ndf, 7, 2, padding='SAME', scope='conv1', reuse=tf.AUTO_REUSE, activation_fn=ops.LeakyReLU)

        conv_cnt = 1
        for depth in layer_depth:
            conv_cnt += 1
            name = "conv%d" % conv_cnt
            x = ops.conv2d(name, x, depth, ksize, 2,
                activation_fn=ops.LeakyReLU,
                normalizer_mode=self.norm_mtd,
                training=self.training,
                reuse=tf.AUTO_REUSE)

        self.disc_out = ops.conv2d("conv%d" % (conv_cnt + 1),
            x, 1, 1, 1,
            activation_fn=None,
            training=self.training,
            reuse=tf.AUTO_REUSE)
        
        print("ImageConditionalDeepDiscriminator shape:")
        print(self.disc_out.get_shape())

        return self.disc_out

class ImageConditionalGenerator(basic.SequentialNN):
    """
    Simulate a style transfer network.
    n_layer : no. of residual blocks recommend to be 3
    """
    def __init__(self, **kwargs):
        super(ImageConditionalGenerator, self).__init__(**kwargs)

    """
    Reflection padding example:
        input = tf.placeholder(tf.float32, [None, 28, 28, 3])
        padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
    """

    def build_inference(self, input):
        def pad_reflect(x, ksize=3, kstride=1):
            pad_width = (ksize // kstride - 1 )// 2
        # assume input to be mask, sketch, image
        # the channel should be 9 dims
        #if len(input) > 1:
        #  x = tf.concat(input, 3, "concat_input_gen")
        #else:
        #    x = input[0]
        x = input

        large_ksize = 9
        p = (large_ksize - 1 )// 2
        input_side = [64, 128, 256]
        output_side = [128, 64]
        residual_dim = 256

        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")
        x = L.conv2d(x, input_side[0], (large_ksize, large_ksize), 1, padding='VALID', scope='conv1', activation_fn=None)
        x = ops.get_norm(x, "conv1/contrib", self.training, tf.AUTO_REUSE)
        x = tf.nn.relu(x)

        # convolution input side
        for idx in range(1, len(input_side), 1):
            name = "conv%d" % (idx + 1)
            # first do reflection padding (hard code to kernel size 3)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            x = L.conv2d(x, input_side[idx], (4, 4), 2, padding='VALID', scope=name, reuse=self.reuse)
            ##x = L.conv2d(x, input_side[idx], (3, 3), padding='VALID')
            ##x = tf.nn.avg_pool(x, (2, 2), [0, 2, 2, 0], padding="VALID")
            # do not need gamma & beta
            x = ops.get_norm(x, name+"/contrib", self.training, self.reuse)
            x = tf.nn.relu(x)
        
        #base_connect = tf.identity(x, "base_connect")
        for idx in range(6):
            name = "Res%d" % (idx + 1)
            shortcut = tf.identity(x)

            # first convolution
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            x = L.conv2d(x, residual_dim, (3, 3), padding='VALID', scope=name+"/conv1")
            x = ops.get_norm(x, name+"/conv1/contrib", self.training, self.reuse)
            x = tf.nn.relu(x)
            # second convolution
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            x = L.conv2d(x, residual_dim, (3, 3), padding='VALID', scope=name+"/conv2")
            x = tf.add(x, shortcut, "residual_block_out")
            #x = ops.get_norm(x, name+"/conv2/contrib", self.training, self.reuse)
            #x = tf.nn.relu(x)
        #x = ops.get_norm(x + base_connect, name+"bridge/contrib", self.training, self.reuse)
        #x = tf.nn.relu(x)
        
        # convolution output side
        for idx in range(len(output_side)):
            name = "conv%d" % (idx + len(input_side) + 1)
            # first do reflection padding (hard code to kernel size 3)
            #x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            
            # do subpixel convolution
            #x = L.conv2d(x, output_side[idx] * 4, (3, 3), padding='VALID', scope=name)
            # pixel shuffle
            #x = tf.depth_to_space(x, 2)
            
            x = L.conv2d_transpose(x, output_side[idx], (4, 4), 2, "SAME", activation_fn=None)
            #x = x[:, 1:-1, 1:-1, :]
            # do not need gamma & beta
            x = ops.get_norm(x, name+"/contrib", self.training, self.reuse)
            x = tf.nn.relu(x)
        
        # output layer
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")
        x = L.conv2d(x, 3, (large_ksize, large_ksize), padding='VALID', scope="conv%d" % (len(input_side) + len(output_side) + 1), activation_fn=None)
        
        x = tf.nn.tanh(x) * 1.1
        self.out = x - tf.nn.relu(x - 1) + tf.nn.relu(-x - 1)

        return self.out

class ImageConditionalEncoder(basic.SequentialNN):
    """
    Simulate a style transfer network.
    n_layer : no. of residual blocks recommend to be 3
    """
    def __init__(self, **kwargs):
        super(ImageConditionalEncoder, self).__init__(**kwargs)

        self.common_length = 4

    """
    Reflection padding example:
        input = tf.placeholder(tf.float32, [None, 28, 28, 3])
        padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
    """

    def build_inference(self):
        def pad_reflect(x, ksize=3, kstride=1):
            p = (ksize - kstride) // 2
            return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

        def residual_block(name, x, ndim, ks=3):
            shortcut = tf.identity(x)
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv1",
                reuse=tf.AUTO_REUSE, activation_fn=None)
            x = ops.get_norm(x, name+"/conv1/" + self.norm_mtd, self.training, tf.AUTO_REUSE)
            x = tf.nn.relu(x)
            # second convolution
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv2",
                reuse=tf.AUTO_REUSE, activation_fn=None)
            x = ops.get_norm(x, name+"/conv2/" + self.norm_mtd, self.training, tf.AUTO_REUSE)
            x = tf.add(x, shortcut, "residual_block_out")
            return x

        x = input

        large_ksize = 9
        p = (large_ksize - 1 )// 2
        input_side = [64, 128, 256]
        output_side = [128, 64]
        residual_dim = 256

        self.norm_mtd = "inst"

        def build_image_feat(x):
            self.conv_cnt = 1
            x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")
            x = L.conv2d(x, input_side[0], (large_ksize, large_ksize), 1, padding='VALID', scope='conv1', activation_fn=None)
            x = ops.get_norm(x, "conv1/contrib", self.training, tf.AUTO_REUSE)
            x = tf.nn.relu(x)

            ### input must be x_real
            for ndf in input_side[1:]:
                self.conv_cnt += 1
                name = 'conv%d' % self.conv_cnt

                x = pad_reflect(x)
                x = L.conv2d(x, ndf, 4, 2, padding='VALID', reuse=tf.AUTO_REUSE,
                    scope=name, activation_fn=None)
                x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, tf.AUTO_REUSE)
                x = tf.nn.relu(x)
            
            # now it is 4x downsampled

            for i in range(self.common_length):
                self.res_cnt += 1
                name = 'Res%d' % self.res_cnt
                x = residual_block(name, x, residual_dim)
            
            return tf.identity(x, "image_feat")
        
        def build_noise_feat(x, map_size=8):
            self.map_size = map_size
            self.map_depth = 128
            # assume 64x64 target
            x = ops.linear("fc1", x,
                        (self.map_size ** 2) * self.map_depth,
                        activation_fn=None,
                        normalizer_mode=None,
                        training=self.training,
                        reuse=tf.AUTO_REUSE)

            x = tf.reshape(x,
                shape=[-1, self.map_size, self.map_size, self.map_depth])
            
            x = ops.get_norm(x,
                name="fc1/"+self.norm_mtd, training=self.training, reuse=tf.AUTO_REUSE)
            
            x = tf.nn.relu(x)
            
            # upsample to 64x64x128:
            map_size = self.map_size
            map_depth = self.map_depth
            for i in range(2):
                self.conv_cnt += 1
                map_size *= 2

                lx = tf.image.resize_images(x, [map_size, map_size])
                x = ops.deconv2d("conv%d" % self.conv_cnt,
                    x, map_depth, 3, 2,
                    activation_fn=tf.nn.relu,
                    normalizer_mode=None,
                    training=self.training,
                    reuse=tf.AUTO_REUSE)
                x = x + lx
                x = ops.get_norm(x,
                    name="conv%d/%s" % (self.conv_cnt, self.norm_mtd),
                    training=self.training,
                    reuse=tf.AUTO_REUSE)

            return x
        
        def build_feat_image(x, ndim=3):
            for i in range(self.common_length):
                self.res_cnt += 1
                name = 'Res%d' % self.res_cnt
                x = residual_block(name, x, residual_dim)
            
            x = ops.get_norm(x, name=self.norm_mtd, training=self.training, reuse=tf.AUTO_REUSE)

            for depth in output_side:
                self.conv_cnt += 1
                x = ops.deconv2d("deconv%d" % self.conv_cnt,
                            x, depth, 3, 2,
                            activation_fn=tf.nn.relu,
                            normalizer_mode=self.norm_mtd,
                            training=self.training,
                            reuse=tf.AUTO_REUSE)

            x = ops.conv2d("deconv%d" % (self.conv_cnt+1),
                        x, ndim, large_ksize, 1,
                        activation_fn=tf.nn.tanh,
                        normalizer_mode=None,
                        training=self.training,
                        reuse=tf.AUTO_REUSE)
            
            return x
        
        def build_feat_noise(x):
            for i in range(self.common_length):
                self.res_cnt += 1
                name = 'Res%d' % self.res_cnt
                x = residual_block(name, x, residual_dim)
            
            x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:])])

            rec_c = ops.linear("fc_c", x, self.c_len, 
                        activation_fn=tf.nn.sigmoid,
                        normalizer_mode=None,
                        training=self.training,
                        reuse=tf.AUTO_REUSE)

            rec_z = ops.linear("fc_z", x, self.z_len,
                        activation_fn=tf.nn.tanh,
                        normalizer_mode=None,
                        training=self.training,
                        reuse=tf.AUTO_REUSE)

            return tf.concat([rec_z, rec_c], axis=1)

        def build_add_noise(image_feat, noise_feat):
            concat_feat = tf.concat([image_feat, noise_feat], axis=3)
            new_feat =  ops.conv2d("conv_add", concat_feat,
                        residual_dim, 3, 1,
                        activation_fn=tf.nn.relu,
                        normalizer_mode=self.norm_mtd,
                        training=self.training,
                        reuse=tf.AUTO_REUSE)
            return new_feat

        with tf.variable_scope("seg_feat"):
            self.res_cnt = self.conv_cnt = 0
            seg_feat_nonoise = build_image_feat(self.seg_input)
            mapsize = seg_feat_nonoise.get_shape().as_list()[2] // 4
            noise_feat = build_noise_feat(self.noise_input, mapsize)
            self.seg_feat = build_add_noise(seg_feat_nonoise, noise_feat)

        with tf.variable_scope("image_feat"):
            self.res_cnt = self.conv_cnt = 0
            self.image_feat = build_image_feat(self.image_input)
            
        with tf.variable_scope("feat_image"):
            self.res_cnt = self.conv_cnt = 0
            self.image_rec = build_feat_image(self.image_feat)
            self.res_cnt = self.conv_cnt = 0
            self.seg_gen = build_feat_image(self.seg_feat)

        with tf.variable_scope("feat_seg"):
            self.res_cnt = self.conv_cnt = 0
            self.image_seg = build_feat_image(self.image_feat, 1)
            self.res_cnt = self.conv_cnt = 0
            self.seg_rec = build_feat_image(self.seg_feat, 1)
        
        # segment -> image; image -> image; segment -> segment; image -> segment
        return self.seg_gen, self.image_rec, self.seg_rec, self.image_seg

class ImageConditionalDeepGenerator2(basic.SequentialNN):
    """
    Simulate a style transfer network.
    n_layer : no. of residual blocks recommend to be 3
    """
    def __init__(self, **kwargs):
        super(ImageConditionalDeepGenerator2, self).__init__(**kwargs)

        self.side_noise = None
        self.gate = tf.placeholder(tf.float32, [], "side_gate")

    """
    Reflection padding example:
        input = tf.placeholder(tf.float32, [None, 28, 28, 3])
        padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
    """

    def build_inference(self, input):
        def pad_reflect(x, ksize=3, kstride=1):
            p = (ksize - kstride) // 2
            return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

        def residual_block(name, x, ndim, ks=3):
            shortcut = tf.identity(x)
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv1",
                reuse=self.reuse, activation_fn=None)
            x = ops.get_norm(x, name+"/conv1/" + self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            # second convolution
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv2",
                reuse=self.reuse, activation_fn=None)
            x = ops.get_norm(x, name+"/conv2/" + self.norm_mtd, self.training, self.reuse)
            x = tf.add(x, shortcut, "residual_block_out")
            return x
        
        def linear(name, x, out_dim):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                w = tf.get_variable("kernel", shape=[x.get_shape()[-1], out_dim])
                return tf.matmul(x, w)

        def conv(name, x, out_dim, ks, kt):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                w = tf.get_variable("kernel", shape=[ks, ks, x.get_shape()[-1], out_dim])
                return tf.nn.conv2d(x, w, [0, kt, kt, 0], padding="SAME")

        x = input

        large_ksize = 7
        ks = 3
        p = (large_ksize - 1 )// 2
        ndf = 32
        self.n_enlarge = 3
        self.norm_mtd = "contrib"

        print("Deep generator shape log:")

        conv_cnt = 0
        res_cnt = 0

        # noise set manually
        if self.side_noise is not None:
            self.side1 = tf.nn.relu(linear("fc_side_map", self.side_noise, 16 * 16 * ndf * 2))
            self.side1 = tf.reshape(self.side1, [-1, 16, 16, ndf * 2])
            self.side1 = L.conv2d(self.side1, ndf * 8, 3,
                activation_fn=tf.nn.relu,
                scope="side_conv1", reuse=tf.AUTO_REUSE)
            
            tmp = self.reuse
            self.reuse = tf.AUTO_REUSE
            # add a residual
            sres_cnt = 0
            for i in range(3):
                sres_cnt += 1
                self.side1 = residual_block("side_residual%d"%sres_cnt, self.side1, ndf * 8)
            self.reuse = tmp
            #self.side2 = tf.nn.tanh(linear("fc_side_feat", self.side_noise, ndf * 8))
            #self.side1 = tf.reshape(self.side1, [-1, 16, 16, 1])
            #self.side2 = tf.reshape(self.side2, [-1, 1, 1, ndf * 8])

        x = pad_reflect(x, large_ksize, 1)
        conv_cnt += 1
        x = L.conv2d(x, ndf, large_ksize, 1, padding='VALID', scope='conv%d'%conv_cnt,
            reuse=self.reuse, activation_fn=None)
        x = ops.get_norm(x, "conv1/" + self.norm_mtd, self.training, self.reuse)
        x = tf.nn.relu(x)
        print(x.get_shape())

        # convolution input side
        mul = 1
        for idx in range(self.n_enlarge):
            mul *= 2
            conv_cnt += 1
            name = 'conv%d'%conv_cnt

            x = pad_reflect(x)
            x = L.conv2d(x, ndf * mul, ks, 2, padding='VALID',
                activation_fn=None,
                scope=name, reuse=self.reuse)
            x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())

        for idx in range(4):
            res_cnt += 1
            name = "Res%d" % res_cnt
            x = residual_block(name, x, ndf * 8)
        
        # Add noise (a gate)
        if self.side_noise is not None:
            #x = L.conv2d(x, ndf * 4, ks, 1, padding="SAME",
            #    activation_fn=tf.nn.relu,
            #    reuse=tf.AUTO_REUSE, scope="side_concat_conv1")
            #x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, tf.AUTO_REUSE)

            x = tf.concat([x, self.side1 * self.gate], axis=3)

            x = L.conv2d(x, ndf * 8, ks, 1, padding="SAME",
                activation_fn=tf.nn.relu,
                reuse=tf.AUTO_REUSE, scope="side_concat_conv2")

            x = ops.get_norm(x, "side_concat/"+self.norm_mtd, self.training, tf.AUTO_REUSE)

        for idx in range(5):
            res_cnt += 1
            name = "Res%d" % res_cnt
            x = residual_block(name, x, ndf * 8)

        # convolution output side
        mul = 8
        for idx in range(self.n_enlarge):
            mul = int(mul/2)
            name = "deconv%d" % (idx + 1)

            # first do reflection padding (hard code to kernel size 3)
            #x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            
            # do subpixel convolution
            #x = L.conv2d(x, ndf * mul, (3, 3), 
            #    padding='VALID',
            #    activation_fn=None,
            #    reuse=self.reuse, scope=name)
            # pixel shuffle
            #x = tf.depth_to_space(x, 2)
            
            x = L.conv2d_transpose(x, ndf * mul, ks, 2, padding="SAME",
                activation_fn=None, scope=name, reuse=self.reuse)
            x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())
        
        # output layer
        x = pad_reflect(x, large_ksize)
        conv_cnt += 1
        x = L.conv2d(x, self.out_dim, large_ksize,
            padding='VALID', scope="conv%d"%conv_cnt,
            reuse=self.reuse, activation_fn=None)

        x = tf.nn.tanh(x)
        self.out = tf.identity(x, "GeneratorOutput")
        print(self.out.get_shape())
        return self.out


class ImageConditionalDeepGenerator(basic.SequentialNN):
    """
    Simulate a style transfer network.
    n_layer : no. of residual blocks recommend to be 3
    """
    def __init__(self, **kwargs):
        super(ImageConditionalDeepGenerator, self).__init__(**kwargs)

    """
    Reflection padding example:
        input = tf.placeholder(tf.float32, [None, 28, 28, 3])
        padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
    """

    def build_inference(self, input):
        def pad_reflect(x, ksize=3, kstride=1):
            p = (ksize - kstride) // 2
            return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

        def residual_block(name, x, ndim, ks=3):
            shortcut = tf.identity(x)
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv1",
                reuse=self.reuse, activation_fn=None)
            x = ops.get_norm(x, name+"/conv1/" + self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            # second convolution
            x = pad_reflect(x, ks)
            x = L.conv2d(x, ndim, ks, padding='VALID', scope=name+"/conv2",
                reuse=self.reuse, activation_fn=None)
            x = ops.get_norm(x, name+"/conv2/" + self.norm_mtd, self.training, self.reuse)
            x = tf.add(x, shortcut, "residual_block_out")
            return x

        # assume input to be mask, sketch, image
        # the channel should be 9 dims
        #if len(input) > 1:
        #  x = tf.concat(input, 3, "concat_input_gen")
        #else:
        #    x = input[0]
        x = input
        
        large_ksize = 7
        ks = 3
        p = (large_ksize - 1 )// 2
        ndf = 32
        self.n_enlarge = 3
        self.norm_mtd = "inst"

        print("Deep generator shape log:")

        conv_cnt = 0
        res_cnt = 0

        x = pad_reflect(x, large_ksize, 1)
        conv_cnt += 1
        x = L.conv2d(x, ndf, large_ksize, 1, padding='VALID', scope='conv%d'%conv_cnt,
            reuse=self.reuse, activation_fn=None)
        x = ops.get_norm(x, "conv1/" + self.norm_mtd, self.training, self.reuse)
        x = tf.nn.relu(x)
        print(x.get_shape())

        # convolution input side
        mul = 1
        for idx in range(self.n_enlarge):
            mul *= 2
            conv_cnt += 1
            name = 'conv%d'%conv_cnt

            x = pad_reflect(x)
            x = L.conv2d(x, ndf * mul, ks, 2, padding='VALID', reuse=self.reuse,
                scope=name, activation_fn=None)
            x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())

        for idx in range(9):
            res_cnt += 1
            name = "Res%d" % res_cnt
            x = residual_block(name, x, ndf * 8)

        # convolution output side
        mul = 8
        for idx in range(self.n_enlarge):
            mul /= 2
            name = "deconv%d" % (idx + 1)

            # first do reflection padding (hard code to kernel size 3)
            #x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            
            # do subpixel convolution
            #x = L.conv2d(x, output_side[idx] * 4, (3, 3), padding='VALID', scope=name)
            # pixel shuffle
            #x = tf.depth_to_space(x, 2)
            
            x = L.conv2d_transpose(x, ndf * mul, ks, 2, "SAME",
                activation_fn=None, scope=name, reuse=self.reuse)
            x = ops.get_norm(x, name+"/"+self.norm_mtd, self.training, self.reuse)
            x = tf.nn.relu(x)
            print(x.get_shape())
        
        # output layer
        x = pad_reflect(x, large_ksize)
        conv_cnt += 1
        x = L.conv2d(x, self.out_dim, large_ksize, padding='VALID', scope="conv%d"%conv_cnt,
            reuse=self.reuse, activation_fn=None)
        #x = ops.get_norm(x, "conv%d"%conv_cnt+"/"+self.norm_mtd, self.training, self.reuse)
        x = tf.nn.tanh(x)
        #x = tf.nn.tanh(x) * 1.1
        #x = x - tf.nn.relu(x - 1) + tf.nn.relu(-x - 1)
        self.out = tf.identity(x, "GeneratorOutput")
        print(self.out.get_shape())
        return self.out