"""
Common loss in neural network training.
Common losses take the form of a function, that is given tensors, return loss tensors.
Common loss should not modify attributes in models.
"""
import tensorflow as tf

def mask_reconstruction_loss(x_real, m_real, s_real, weight=1.0):
    """
    """
    loss = tf.reduce_mean(tf.abs(x_real - s_real) * m_real)
    loss = tf.multiply(loss, weight, "MaskRec")
    loss_sum = tf.summary.scalar("MaskRec", loss)
    return loss, loss_sum

def reconstruction_loss(x_rec, x_real, weight=1.0, name="Rec"):
    rec_loss = weight * tf.reduce_mean(tf.square(x_rec - x_real), name=name)
    loss_sum = tf.summary.scalar(name, rec_loss)
    return rec_loss, loss_sum

def regulation_loss(model, weight=1e-3, add_summary=False):
    """
    Currently L2 regularization.
    """
    with tf.name_scope("L2_regularization") as nsc:
        reg_loss = tf.multiply(sum([tf.reduce_sum(v ** 2) for v in model.vars]), weight, name=nsc)

    if add_summary:
        reg_sum = tf.summary.scalar("L2 Regularization Loss", reg_loss)
        return reg_loss, reg_sum
    return reg_loss, None

def gradient_penalty(model, X, X_p, add_summary=False):
    """
    Return a gradient penalty for a model. The GP is computed between D(X) and D(X_p).
    Args:
    X   :   Base data
    X_p :   A transformed data.
    """

    batch_size = tf.shape(X)[0]

    alpha = tf.reshape(tf.random_uniform(
        shape=[batch_size],
        minval=0.,
        maxval=1.
    ), [-1, 1, 1, 1])

    #alpha = tf.random_uniform([], minval=0., maxval=1.)

    with tf.name_scope("GP_diff"):
        differences = X_p - X
        interpolates = X + alpha * differences

    model.set_reuse()
    disc_interp = model(interpolates)[0]
    
    #disc_interp = tf.reduce_sum(disc_interp)
    #disc_interp_ = tf.reshape(disc_interp, [-1])

    with tf.name_scope("GP_grad"):
        gradients = tf.gradients(disc_interp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    
    gradient_penalty = tf.reduce_mean(tf.square(slopes-1.), name="GP_reduce_mean")

    return gradient_penalty