import sys
sys.path.insert(0, ".")
import tensorflow as tf
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import skimage.io as io
from lib.utils import load_mnist_4d, save_batch_img
from test.analysis import *
from lib.ops import LeakyReLU, random_truncate_normal

base_dir = "test/expr/gan/"
DTYPE = tf.float32
N_SAMPLE = 4096
GT = 2000
NGT = 1
# set up
gen_data = random_truncate_normal((N_SAMPLE, 784), 1, 0)
global_iter = 0
rec_loss_real = []
rec_loss_fake = []
rec_loss_gen = []
logfile = open("test/expr/log.txt", "w")

def get_video(base_dir, name_types):
    for t in name_types:
        type_str = t + ".png"
        cmd = "ffmpeg -f image2 -r 20 -i %s -vcodec mpeg4 expr/%s.mp4"
        os.system(cmd)

def linear(x, vars):
    x = tf.matmul(x, vars[0]) + vars[1]
    return x

def build_net(x, vars):
    for i in range(len(vars)):
        x = linear(x, vars[i])
        if i != len(vars) - 1:
            x = LeakyReLU(x)
    # x = tf.nn.sigmoid(x)
    # output is logits
    return x

def build_convex(x, vars):
    return tf.nn.tanh(linear(x, vars[i]))

def from_np(nparr):
    return tf.cast(nparr, DTYPE)

def get_vars(dims):
    vars = []
    for l in range(len(dims)-1):
        w = tf.get_variable("fc_w_%d" % l, shape=(dims[l], dims[l+1]), initializer=tf.contrib.slim.xavier_initializer(dtype=DTYPE))
        b = tf.get_variable("fc_b_%d" % l, initializer=from_np(np.zeros((dims[l+1]))), dtype=DTYPE)
        vars.append((w, b))
    return vars

def get_update_op(vars, grad, lr):
    update_ops = []
    for i in range(len(grad)):
        update_ops.append(tf.assign_add(vars[i], -lr * grad[i]))
    return update_ops

def flatten_list(npylist):
    """
    flatten a list of np array
    """
    res = []
    for item in npylist:
        res.extend(item.reshape(-1).tolist())
    return np.array(res)

def fltlist2net(x, flt_vars):
    """
    Args:
    flt_vars:   Assumes to be list
    """
    vars = []
    for i in range(len(flt_vars)//2):
        v1, v2 = tf.Variable(flt_vars[2*i]), tf.Variable(flt_vars[2*i+1])
        vars.append((v1, v2))
    
    return build_net(x, vars)

def main():
    lr = tf.placeholder(DTYPE, shape=[], name="learning_rate")
    raw_real_data = tf.placeholder(tf.float64, [None, 784], "real_data")
    real_data = tf.cast(raw_real_data, DTYPE)
    raw_fake_data = tf.placeholder(tf.float64, [None, 784], "real_data")
    fake_data = tf.cast(raw_fake_data, DTYPE)

    trX, teX, trY, teY = load_mnist_4d("data/MNIST"); del teX, teY
    trX = trX.reshape(trX.shape[0], -1) / 127.5 - 1

    # build net
    with tf.variable_scope("disc", reuse=tf.AUTO_REUSE):
        vars = get_vars([784, 128, 1])
    flt_vars = []
    for vp in vars:
        flt_vars.extend(vp)
    disc_real = build_net(real_data, vars)
    disc_fake = build_net(fake_data, vars)

    # build loss
    raw_gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.ones_like(disc_fake)), name="raw_gen_cost")

    raw_disc_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros_like(disc_fake)), name="raw_disc_cost_fake")

    raw_disc_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones_like(disc_real)), name="raw_disc_cost_real")

    #disc_cost = raw_disc_cost_real + raw_disc_cost_fake
    #gen_cost = raw_gen_cost

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake) 

    # compute gradient

    grad_input = tf.gradients(gen_cost, raw_fake_data)[0]
    grad_disc = tf.gradients(disc_cost * 1e4, flt_vars)
    # notice : grad disc is 1e4 enlarged

    # update op

    #update_op = get_update_op(flt_vars, grad_disc, lr)
    #train_disc_op = tf.train.AdamOptimizer(lr, 0, 0).minimize(disc_cost, var_list=flt_vars)
    train_disc_op = tf.train.GradientDescentOptimizer(lr).minimize(disc_cost, var_list=flt_vars)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ## initial status

    def basic_statistic(ifhist=False):
        global rec_loss_fake, rec_loss_real, rec_loss_gen

        fetches = [disc_cost, gen_cost]
        sample_disc_cost, sample_gen_cost = sess.run(fetches,
            {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})
        rec_loss_fake.append(sample_disc_cost)
        rec_loss_gen.append(sample_gen_cost)

        print("iter %d, disc_cost %.3f, gen_cost %.3f" % (global_iter, sample_disc_cost, sample_gen_cost))
        logfile.write("iter %d, disc_cost %.3f, gen_cost %.3f" % (global_iter, sample_disc_cost, sample_gen_cost)) 

        """
        fetches = [disc_real, disc_fake, raw_disc_cost_real, raw_disc_cost_fake, raw_gen_cost]
        if ifhist:
            fetches.extend([grad_input * 1e4, grad_disc])
            sample_disc_real, sample_disc_fake, sample_real_cost, sample_fake_cost, sample_gen_cost, sample_grad_input, sample_grad_disc = sess.run(fetches, {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})
            hist(sample_disc_real, base_dir + "hist_real_%d"        %   global_iter)
            hist(sample_disc_fake, base_dir + "hist_fake_%d"        %   global_iter)
            hist(flatten_list(sample_grad_input),base_dir + "hist_grad_input_%d"%global_iter, (-1, 1))
            hist(flatten_list(sample_grad_disc),base_dir + "hist_grad_disc_%d"%global_iter, (-1, 1))

        else:
            sample_disc_real, sample_disc_fake, sample_real_cost, sample_fake_cost, sample_gen_cost = sess.run(fetches, {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})

        rec_loss_real.append(sample_real_cost)
        rec_loss_fake.append(sample_fake_cost)
        rec_loss_gen.append(sample_gen_cost)

        print("iter %d, real_cost %.3f, fake_cost %.3f, gen_cost %.3f" % (global_iter, sample_real_cost, sample_fake_cost, sample_gen_cost))
        logfile.write("iter %d, real_cost %.3f, fake_cost %.3f, gen_cost %.3f\n" % (global_iter, sample_real_cost, sample_fake_cost, sample_gen_cost))    
        """

    def train_op(learning_rate=0.001):
        global global_iter
        sess.run(train_disc_op, {
            raw_real_data : trX[:N_SAMPLE],
            raw_fake_data : gen_data,
            lr : learning_rate
            })
        global_iter += 1

    def update_gen_data(learning_rate=1):
        global gen_data
        rg = 0
        for i in range(NGT):
            g = sess.run([grad_input], {
                raw_fake_data : gen_data
            })[0] * learning_rate

            gen_data -= g

            rg = g[0,:] + rg

        return rg

    def one_iter(learning_rate=0.01, do_statis=False):
        train_op(learning_rate)
        basic_statistic(False)
        g = update_gen_data(GT)#.sum(0).reshape(28, 28)
        logfile.write("Iter%d:\n" % global_iter)
        logfile.write("grad input %.4f %.4f %.4f\n" % (g.max(), g.min(), g.mean()))
        logfile.write("gen data %.4f %.4f %.4f\n" % (gen_data[:512].max(), gen_data[:512].min(), gen_data[:512].mean()))

        if do_statis:
            basic_statistic(False)
            #save_batch_img(trX.reshape(-1, 28, 28, 1), base_dir + "real_%d.png" % global_iter)
            #save_batch_img(gen_data[:16].reshape(-1, 28, 28, 1), base_dir + "sample_%d.png" % global_iter)

        logfile.flush()

    basic_statistic()
    try:
        for i in range(350):
            one_iter(0.001, True)
            if i % 10 == 0:
                var_np = sess.run(flt_vars)
                np.save("test/expr/weight/%05d" % i, var_np)

            if i % 100 == 0:
                plot(rec_loss_fake, "test/expr/loss_fake_%d"% NGT)
                plot(rec_loss_real, "test/expr/loss_real_%d"% NGT)
                plot(rec_loss_gen,  "test/expr/loss_gen_%d" % NGT)

    except KeyboardInterrupt:
        print("Get Interrupt")
        plt.close()

    plot(rec_loss_fake, "test/expr/loss_fake_%d"% NGT)
    plot(rec_loss_real, "test/expr/loss_real_%d"% NGT)
    plot(rec_loss_gen,  "test/expr/loss_gen_%d" % NGT)
    logfile.close()

def main_convex():
    print("Convex game")

    lr = tf.placeholder(DTYPE, shape=[], name="learning_rate")
    raw_real_data = tf.placeholder(tf.float64, [None, 784], "real_data")
    real_data = tf.cast(raw_real_data, DTYPE)
    raw_fake_data = tf.placeholder(tf.float64, [None, 784], "real_data")
    fake_data = tf.cast(raw_fake_data, DTYPE)

    trX, teX, trY, teY = load_mnist_4d("data/MNIST"); del teX, teY
    trX = trX.reshape(trX.shape[0], -1) / 127.5 - 1

    # build net
    with tf.variable_scope("disc", reuse=tf.AUTO_REUSE):
        vars = get_vars([784, 1])
    flt_vars = []
    for vp in vars:
        flt_vars.extend(vp)
    disc_real = build_net(real_data, vars)
    disc_fake = build_net(fake_data, vars)

    # build loss
    raw_gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.ones_like(disc_fake)), name="raw_gen_cost")

    raw_disc_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros_like(disc_fake)), name="raw_disc_cost_fake")

    raw_disc_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones_like(disc_real)), name="raw_disc_cost_real")

    #disc_cost = raw_disc_cost_real + raw_disc_cost_fake
    #gen_cost = raw_gen_cost

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake) 

    # compute gradient

    grad_input = tf.gradients(gen_cost, raw_fake_data)[0]
    grad_disc = tf.gradients(disc_cost * 1e4, flt_vars)
    # notice : grad disc is 1e4 enlarged

    # update op

    #update_op = get_update_op(flt_vars, grad_disc, lr)
    #train_disc_op = tf.train.AdamOptimizer(lr, 0, 0).minimize(disc_cost, var_list=flt_vars)
    train_disc_op = tf.train.GradientDescentOptimizer(lr).minimize(disc_cost, var_list=flt_vars)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ## initial status

    def basic_statistic(ifhist=False):
        global rec_loss_fake, rec_loss_real, rec_loss_gen

        fetches = [disc_cost, gen_cost]
        sample_disc_cost, sample_gen_cost = sess.run(fetches,
            {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})
        rec_loss_fake.append(sample_disc_cost)
        rec_loss_gen.append(sample_gen_cost)

        print("iter %d, disc_cost %.3f, gen_cost %.3f" % (global_iter, sample_disc_cost, sample_gen_cost))
        logfile.write("iter %d, disc_cost %.3f, gen_cost %.3f" % (global_iter, sample_disc_cost, sample_gen_cost)) 

        """
        fetches = [disc_real, disc_fake, raw_disc_cost_real, raw_disc_cost_fake, raw_gen_cost]
        if ifhist:
            fetches.extend([grad_input * 1e4, grad_disc])
            sample_disc_real, sample_disc_fake, sample_real_cost, sample_fake_cost, sample_gen_cost, sample_grad_input, sample_grad_disc = sess.run(fetches, {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})
            hist(sample_disc_real, base_dir + "hist_real_%d"        %   global_iter)
            hist(sample_disc_fake, base_dir + "hist_fake_%d"        %   global_iter)
            hist(flatten_list(sample_grad_input),base_dir + "hist_grad_input_%d"%global_iter, (-1, 1))
            hist(flatten_list(sample_grad_disc),base_dir + "hist_grad_disc_%d"%global_iter, (-1, 1))

        else:
            sample_disc_real, sample_disc_fake, sample_real_cost, sample_fake_cost, sample_gen_cost = sess.run(fetches, {raw_real_data : trX[:N_SAMPLE], raw_fake_data : gen_data})

        rec_loss_real.append(sample_real_cost)
        rec_loss_fake.append(sample_fake_cost)
        rec_loss_gen.append(sample_gen_cost)

        print("iter %d, real_cost %.3f, fake_cost %.3f, gen_cost %.3f" % (global_iter, sample_real_cost, sample_fake_cost, sample_gen_cost))
        logfile.write("iter %d, real_cost %.3f, fake_cost %.3f, gen_cost %.3f\n" % (global_iter, sample_real_cost, sample_fake_cost, sample_gen_cost))    
        """

    def train_op(learning_rate=0.001):
        global global_iter
        sess.run(train_disc_op, {
            raw_real_data : trX[:N_SAMPLE],
            raw_fake_data : gen_data,
            lr : learning_rate
            })
        global_iter += 1

    def update_gen_data(learning_rate=1):
        global gen_data
        rg = 0
        for i in range(NGT):
            g = sess.run([grad_input], {
                raw_fake_data : gen_data
            })[0] * learning_rate

            gen_data -= g

            rg = g[0,:] + rg

        return rg

    def one_iter(learning_rate=0.01, do_statis=False):
        train_op(learning_rate)
        basic_statistic(False)
        g = update_gen_data(GT)#.sum(0).reshape(28, 28)
        logfile.write("Iter%d:\n" % global_iter)
        logfile.write("grad input %.4f %.4f %.4f\n" % (g.max(), g.min(), g.mean()))
        logfile.write("gen data %.4f %.4f %.4f\n" % (gen_data[:512].max(), gen_data[:512].min(), gen_data[:512].mean()))

        if do_statis:
            basic_statistic(False)
            #save_batch_img(trX.reshape(-1, 28, 28, 1), base_dir + "real_%d.png" % global_iter)
            #save_batch_img(gen_data[:16].reshape(-1, 28, 28, 1), base_dir + "sample_%d.png" % global_iter)

        logfile.flush()

    basic_statistic()
    try:
        for i in range(500):
            one_iter(0.001, True)
            if i % 10 == 0:
                var_np = sess.run(flt_vars)
                np.save("test/expr/weight/%05d" % i, var_np)

            if i % 100 == 0:
                plot(rec_loss_fake, "test/expr/loss_fake_%d"% NGT)
                plot(rec_loss_real, "test/expr/loss_real_%d"% NGT)
                plot(rec_loss_gen,  "test/expr/loss_gen_%d" % NGT)

    except KeyboardInterrupt:
        print("Get Interrupt")
        plt.close()

    plot(rec_loss_fake, "test/expr/loss_fake_%d"% NGT)
    plot(rec_loss_real, "test/expr/loss_real_%d"% NGT)
    plot(rec_loss_gen,  "test/expr/loss_gen_%d" % NGT)
    logfile.close()

if __name__ == "__main__":
    main()
    #main_convex()