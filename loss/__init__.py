"""
GAN loss module
"""

import tensorflow as tf

from lib import ops
import numpy as np
from loss.common_loss import gradient_penalty

def classifier_loss(gen_model, disc_model, x_real, c_label, c_noise, weight=1.0):
    print("=> classifier loss build")

    real_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_model.real_cls_logits,
        labels=c_label), name="real_cls_reduce_mean")

    fake_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_model.fake_cls_logits,
        labels=c_noise), name='fake_cls_reduce_mean')

    gen_model.cost += weight * fake_cls
    disc_model.cost += weight * (real_cls + fake_cls)

    print("=> classifier loss summary")
    
    with tf.device("/device:CPU:0"):
        real_cls_sum = tf.summary.scalar("real_cls", real_cls)
        fake_cls_sum = tf.summary.scalar("fake_cls", fake_cls)

    disc_model.sum_op.extend([real_cls_sum, fake_cls_sum])

def wass_loss(gen_model, disc_model, gen_input, x_real, adv_weight=1.0, gp_weight=10.0):
    print("=> Wassertain loss build")
    gp_loss = gp_weight * gradient_penalty(disc_model, x_real, gen_model.x_fake)
    disc_model.cost += gp_loss

    gen_model.cost += -adv_weight * tf.reduce_mean(disc_model.disc_fake)
    wass_dist = tf.reduce_mean(disc_model.disc_fake) - tf.reduce_mean(disc_model.disc_real)
    disc_model.cost += adv_weight * wass_dist

    with tf.name_scope(disc_model.name):
        disc_model.sum_op.extend([
            tf.summary.scalar("distance", wass_dist),
            tf.summary.scalar("gradient penalty", gp_loss)
            ])

def func_naive_ganloss(disc_fake_out, disc_real_out, adv_weight=1, lsgan=False, name="NaiveGAN"):
    """
    Require the model be built first
    """
    print("=> get_naive_ganloss")
    raw_gen_cost = raw_disc_cost_fake = raw_disc_cost_real = 0
    if lsgan:
        raw_gen_cost = tf.reduce_mean(tf.square(
            disc_fake_out - 1),
            name="raw_gen_cost")

        raw_disc_cost_fake = tf.reduce_mean(tf.square(
            disc_fake_out),
            name="raw_disc_cost_fake")

        raw_disc_cost_real = tf.reduce_mean(tf.square(
            disc_real_out - 1),
            name="raw_disc_cost_real")
    else:
        raw_gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake_out,
            labels=tf.ones_like(disc_fake_out)), name="raw_gen_cost")
        
        raw_disc_cost_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake_out,
            labels=tf.zeros_like(disc_fake_out)), name="raw_disc_cost_fake")

        raw_disc_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real_out,
            labels=tf.ones_like(disc_real_out)), name="raw_disc_cost_real")

    disc_cost = tf.multiply(adv_weight,
            raw_disc_cost_fake + raw_disc_cost_real,
            name="disc_cost")
    gen_cost = tf.multiply(raw_gen_cost, adv_weight, name="gen_cost")

    disc_real_sum = tf.summary.scalar("DiscRealRaw", raw_disc_cost_real)
    disc_fake_sum = tf.summary.scalar("DiscFakeRaw", raw_disc_cost_fake)
    gen_cost_sum = tf.summary.scalar("GenRaw", raw_gen_cost)
    
    return gen_cost, disc_cost, [gen_cost_sum], [disc_fake_sum, disc_real_sum]

def naive_ganloss(gen_model, disc_model, adv_weight=1, lsgan=False, name="NaiveGAN"):
    gen_cost, disc_cost, gen_sum, disc_sum = func_naive_ganloss(
        disc_model.disc_fake, disc_model.disc_real, adv_weight=adv_weight, lsgan=lsgan, name=name)
    gen_model.cost += gen_cost
    gen_model.sum_op.extend(gen_sum)
    disc_model.cost += disc_cost
    disc_model.sum_op.extend(disc_sum)

def dragan_loss(gen_model, disc_model, x_real, adv_weight=1.0, gp_weight=1.0):
    print("=> dragan loss build")

    naive_ganloss(gen_model, disc_model, adv_weight)

    # perturbate real input
    # get the variance of total batch
    with tf.name_scope("preturb_real_data") as nsc:
        real_var = tf.reduce_sum(tf.nn.moments(tf.reshape(x_real, [-1]), [0])[1])
        real_std = tf.sqrt(real_var)
        # perturb the data
        dragan_perturb_data = tf.add(x_real, 0.5 * real_std * tf.random_uniform(tf.shape(x_real), name=nsc))

    raw_gp = gradient_penalty(disc_model, x_real, dragan_perturb_data)
    disc_model.cost += gp_weight * raw_gp
    disc_model.sum_op.append(tf.summary.scalar("gradient penalty (dra)", raw_gp))

def hinge_loss(gen_model, disc_model, adv_weight=1.0):
    print("=> Build Hinge loss")
    raw_gen_cost = - tf.reduce_mean(disc_model.disc_fake)
    raw_disc_cost = tf.reduce_mean(tf.nn.relu(1 - disc_model.disc_real))
    raw_disc_cost += tf.reduce_mean(tf.nn.relu(1 + disc_model.disc_fake))
    
    gen_model.cost += raw_gen_cost * adv_weight
    disc_model.cost += raw_disc_cost * adv_weight
    with tf.device("/device:CPU:0"):
        gen_model.sum_op.append(tf.summary.scalar("GenRaw", raw_gen_cost))
        disc_model.sum_op.append(tf.summary.scalar("DiscRaw", raw_disc_cost))

def diverse_distribution(ps):
    """
    ps: (batch, size, size, k), diverse at k
    """
    shape = tf.shape(ps)
    ps = tf.transpose(ps, [0, 3, 1, 2])
    # (batch, k, size * size)
    s = tf.reshape(ps, [shape[0], shape[3], shape[1] * shape[2]])
    #s = tf.Print(s, ["s", tf.reduce_sum(s[0], [1])])
    s = tf.sqrt(s + 1e-8)

    ss = tf.matmul(s, tf.transpose(s, [0, 2, 1]))
    shape = tf.shape(ss)
    #ss = tf.Print(ss, [tf.shape(ss)])
    #ss = tf.Print(ss, [ss[0,0,0],ss[1,0,1],ss[2,1,0],ss[3,1,1]])

    eye = tf.eye(shape[1], shape[2], (shape[0],))
    loss = tf.reduce_sum(tf.square(eye - ss)) / tf.cast(shape[0], tf.float32)
    loss_sum = tf.summary.scalar("diverse_loss", loss)

    return loss, loss_sum

def cosine_diverse_distribution(ps):
    """
    ps: (batch, size, size, k), diverse at k
    """
    shape = tf.shape(ps)
    ps = tf.transpose(ps, [0, 3, 1, 2])

    # (batch, k, size * size)
    s = tf.reshape(ps, [shape[0], shape[3], shape[1] * shape[2]])
    s = tf.nn.l2_normalize(s, dim=2)
    ss = tf.matmul(s, tf.transpose(s, [0, 2, 1]))

    shape = tf.shape(ss)
    eye = tf.eye(shape[1], shape[2], (shape[0],))

    loss = tf.reduce_mean(tf.square(eye - ss))
    loss_sum = tf.summary.scalar("cosine diverse loss", loss)

    return loss, loss_sum