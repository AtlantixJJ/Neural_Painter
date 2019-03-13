"""
Monitor the process of basic GAN training. But I suggest to use simple trainer, because this trainer is very complicated, uneccessarily.
"""
import tensorflow as tf

from trainer.base_trainer import BaseTrainer
import numpy as np
from lib import ops, utils, files, cache, adabound
import skimage.io as io
import time, tqdm

class BaseGANTrainer(BaseTrainer):
    """
    Monitor the process of basic GAN training.
    """

    def __init__(self, gen_model, disc_model, gen_input, x_real, label, sample_method=None, **kwargs):
        super(BaseGANTrainer, self).__init__(**kwargs)

        self.g_lr = tf.placeholder(tf.float32, [], "g_lr")
        self.d_lr = tf.placeholder(tf.float32, [], "d_lr")

        if sample_method is None:
            self.use_cache = False
        else:
            self.sample_cost, self.sample_sum, self.sample_input = sample_method
            self.cache = cache.SampleCache(self.batch_size, 100)
            self.use_cache = True
        
        self.gen_model = gen_model
        self.disc_model = disc_model

        if self.FLAGS.cgan:
            self.z_noise, self.c_noise = gen_input
            self.c_len = gen_input[1].get_shape().as_list()[-1]
            self.z_len = gen_input[0].get_shape().as_list()[-1]
            self.c_label = label
            self.c_shape = (self.batch_size, self.c_len)
        else:
            self.z_noise = gen_input
            self.z_len = gen_input.get_shape().as_list()[-1]
        
        self.x_real = x_real
        self.z_shape = (self.batch_size, self.z_len)

    def reset_train_state(self):
        super(BaseGANTrainer, self).reset_train_state()

        self.sample_iter = 0
        self.disc_tot_iter, self.gen_tot_iter = 0, 0
        self.g_loss, self.d_loss = 0.0, 0.0

        self.run_time = 0
        self.total_time = 0

        self.make_fixed_feed()

    def reload(self):
        self.disc_model.load_from_npz(self.FLAGS.rpath + "_disc.npz", self.sess)
        self.gen_model.load_from_npz(self.FLAGS.rpath + "_gen.npz", self.sess)
        self.is_init = True

    def build_train_op(self):
        """
        Not to be modified in common cases.
        """
        self.gen_model.build_train_op(self.g_lr)
        self.disc_model.build_train_op(self.d_lr)
        if self.use_cache:
            with tf.control_dependencies(self.gen_model.update_ops):
                """
                self.sample_train_op = adabound.AdaBoundOptimizer(
                    learning_rate=4e-4,
                    final_lr=1e-3, beta1=0.9, beta2=0.999,
                    gamma=1e-3, epsilon=1e-8).minimize(self.sample_cost, var_list=self.disc_model.vars, colocate_gradients_with_ops=True)
                """
                self.sample_train_op = tf.train.AdamOptimizer(
                    learning_rate=self.lr,
                    beta1=0.5,
                    beta2=0.9).minimize(self.sample_cost,
                        var_list=self.disc_model.vars,
                        colocate_gradients_with_ops=True)

    def make_fixed_feed(self):
        self.fixed_feed = {
            self.gen_model.keep_prob : 1.0,
            self.disc_model.keep_prob : 1.0,
            self.gen_model.training: False,
            self.disc_model.training: False
        }

        self.fixed_noise = ops.get_random("normal", self.z_shape)
        
        if self.FLAGS.cgan:
            if self.c_shape[1] == 34: self.fixed_cond = ops.get_random("getchu_continuous", self.c_shape)
            else: self.fixed_cond = ops.get_random("boolean", self.c_shape)

            self.fixed_feed.update({
                self.z_noise: self.fixed_noise,
                self.c_noise: self.fixed_cond
            })
        else:
            self.fixed_feed.update({
                self.z_noise: self.fixed_noise
            })

    def schedule_lr(self):
        def get_lr(start_lr):
            if self.global_iter < self.FLAGS.dec_iter:
                cur_lr = start_lr
            elif self.global_iter < self.FLAGS.num_iter:
                cur_lr = utils.linear_interpolate(
                    x_bg=self.FLAGS.dec_iter,
                    x_ed=self.FLAGS.num_iter,
                    y_bg=start_lr,
                    y_ed=start_lr / 10,
                    cur_x=self.global_iter)
            else:
                cur_lr = start_lr
            return cur_lr
            
        self.feed.update({
            self.g_lr : get_lr(self.FLAGS.g_lr),
            self.d_lr : get_lr(self.FLAGS.d_lr)
        }) 

    def make_feed(self, sample):
        self.x_real_img = np.array(sample[0])
        
        if self.FLAGS.cgan:
            self.c_real_label = np.array(sample[1])
            self.feed.update({
                self.c_label: self.c_real_label,
            })

        self.schedule_lr()
        self.resample_feed()

        self.feed.update({
            self.x_real : self.x_real_img,
            self.disc_model.training    : True,
            self.gen_model.training     : True
        })
    
    def resample_feed(self):
        """
        random z_noise
        """
        z = ops.get_random("normal", self.z_shape)
        self.feed.update({self.z_noise : z})

        if self.FLAGS.cgan:
            if self.c_shape[1] == 34: c = ops.get_random("getchu_continuous", self.c_shape)
            else: c = ops.get_random("boolean", self.c_shape)
            self.feed.update({self.c_noise : c})

    def train_epoch(self):
        self.last_gen_loss, self.last_disc_loss = 0.0, 0.0

        def inc_log(sum_, inc_global=True):
            self.summary_writer.add_summary(sum_, self.global_iter)
            if inc_global:
                self.global_iter = self.global_iter + 1

        print("=> Epoch %d" % self.epoch_count)

        start_time = time.clock()

        self.dataloader.reset()
        self.epoch_count += 1
        LEN = len(self.dataloader)
        for i_ in tqdm.tqdm(range(LEN)):
            sample = self.dataloader[i_]
            if len(sample[0].shape) == 5:
                s = sample[0].shape
                sample[0] = sample[0].reshape([s[0] * s[1], s[2], s[3], s[4]])
                s = sample[1].shape
                sample[1] = sample[1].reshape([s[0] * s[1], s[2]])
            if sample[0].shape[0] < self.batch_size:
                print("=> Skip incomplete batch")
                continue
            self.make_feed(sample)
            last_iter = self.global_iter

            t0 = time.clock()

            for j in range(self.FLAGS.disc_iter):
                _, sum_ = self.sess.run([
                    self.disc_model.train_op,
                    self.disc_model.sum_op], self.feed)
                inc_log(sum_, True)
                self.disc_tot_iter += 1
                self.resample_feed()

                if self.global_iter > 2000 and self.use_cache:
                    self.feed.update({
                        self.sample_input : self.cache.get()[0],
                        self.cache.batch_full : len(self.cache.database[0])
                        })
                    _, sum_, sum__ = self.sess.run([
                        self.sample_train_op,
                        self.sample_sum,
                        self.cache.batch_full_sum], self.feed)
                    self.summary_writer.add_summary(sum_, self.sample_iter)
                    self.summary_writer.add_summary(sum__, self.sample_iter)
                    self.disc_tot_iter += 1
                    self.sample_iter += 1

            for j in range(self.FLAGS.gen_iter):
                fake_sample, _, sum_ = self.sess.run([
                    self.gen_model.x_fake,
                    self.gen_model.train_op, 
                    self.gen_model.sum_op], self.feed)
                if np.random.rand() < 1.0 / self.batch_size / 2 and self.use_cache:
                    self.cache.add([fake_sample], self.global_iter)
                inc_log(sum_, True)
                self.gen_tot_iter += 1
                self.resample_feed()
            
            self.run_time += time.clock() - t0
            self.total_time = time.clock() - start_time

            self.maybe_intersum(last_iter)
            self.maybe_save(last_iter)

    def maybe_save(self, last_iter):
        int_iter_num = self.global_iter // self.FLAGS.save_iter
        if int_iter_num > last_iter // self.FLAGS.save_iter:
            self.gen_model.get_trainable_variables()
            self.disc_model.get_trainable_variables()
            
            self.saver.save(self.sess, '%s/sess' % self.FLAGS.train_dir,
                global_step=self.global_iter)
            
            self.gen_model.save_to_npz(self.FLAGS.train_dir + ("/gen_%02d.npz" % int_iter_num), self.sess)
            self.disc_model.save_to_npz(self.FLAGS.train_dir + ("/disc_%02d.npz" % int_iter_num), self.sess)

    def maybe_intersum(self, last_iter):
        """
        if self.global_iter // 10 > last_iter // 10:
            gen_imgs = self.sess.run([self.gen_model.x_fake], self.fixed_feed)[0]
            utils.save_batch_img(gen_imgs, self.TFLAGS['train_dir']+"/gen%d.png" % (self.global_iter // 10))
        """
        
        int_iter_num = self.global_iter // 1000
        if int_iter_num > last_iter // 1000 or self.req_inter_sum:
            print("=> Iter[%d], Gen[%d], Disc[%d]" % (self.global_iter, self.gen_tot_iter, self.disc_tot_iter) )
            print("=> [%05d] Run ratio: %f" % (self.global_iter, self.run_time / self.total_time))

            self.fixed_feed.update({
                self.x_real : self.x_real_img
            })

            if self.use_cache:
                self.fixed_feed.update({
                    self.sample_input : self.cache.get()[0]
                })
                
            if self.FLAGS.cgan:
                self.fixed_feed.update({
                self.c_label : self.c_real_label
                })

            inter_sum = self.sess.run(
                [self.int_sum_op],
                self.fixed_feed)[0]

            self.summary_writer.add_summary(inter_sum, int_iter_num)
            self.req_inter_sum = False
