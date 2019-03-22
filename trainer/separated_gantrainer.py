"""
Separate disc real training, disc fake training and gen training.
"""
import tensorflow as tf

from trainer.base_gantrainer import BaseGANTrainer
import numpy as np
from lib import ops, utils, files, cache, adabound
import skimage.io as io
import time, tqdm

class SeparatedGANTrainer(BaseGANTrainer):
    def __init__(self, gen_model, disc_model, gen_input, x_real, label, sample_method=None, **kwargs):
        super(SeparatedGANTrainer, self).__init__(gen_model, disc_model, gen_input, x_real, label, sample_method=None, **kwargs)
    
    def build_train_op(self):
        """
        Not to be modified in common cases.
        """
        self.gen_model.vars = [v for v in tf.trainable_variables() if self.gen_model.name in v.name]
        self.disc_model.vars = [v for v in tf.trainable_variables() if self.disc_model.name in v.name]
        self.gen_model.sum_op = tf.summary.merge(self.gen_model.sum_op)
        self.disc_model.sum_op = tf.summary.merge(self.disc_model.sum_op)

        self.d_real_op = tf.train.AdamOptimizer(
                learning_rate=self.d_lr,
                beta1=0.,
                beta2=0.9).minimize(self.disc_model.disc_real_loss, var_list=self.disc_model.vars, colocate_gradients_with_ops=True)
        self.d_fake_op = tf.train.AdamOptimizer(
                learning_rate=self.d_lr,
                beta1=0.,
                beta2=0.9).minimize(self.disc_model.disc_fake_loss, var_list=self.disc_model.vars, colocate_gradients_with_ops=True)
        self.g_train_op = tf.train.AdamOptimizer(
                learning_rate=self.g_lr,
                beta1=0.,
                beta2=0.9).minimize(self.gen_model.cost, var_list=self.gen_model.vars, colocate_gradients_with_ops=True)

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
                self.sess.run([self.d_real_op,self.d_fake_op], self.feed)
                sum_ = self.sess.run(self.disc_model.sum_op, self.feed)
                inc_log(sum_, True)
                self.disc_tot_iter += 1
                self.resample_feed()

            for j in range(self.FLAGS.gen_iter):
                _, sum_ = self.sess.run([
                    self.g_train_op, 
                    self.gen_model.sum_op], self.feed)

                inc_log(sum_, True)
                self.gen_tot_iter += 1
                self.resample_feed()
            
            self.run_time += time.clock() - t0
            self.total_time = time.clock() - start_time

            self.maybe_intersum(last_iter)
            self.maybe_save(last_iter)

 