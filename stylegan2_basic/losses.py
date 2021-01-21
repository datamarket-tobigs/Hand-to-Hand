import tensorflow as tf
import os
import numpy as np

def l1_loss(feature_1, feature_2, axis):
    l1_diff = tf.abs(feature_1 - feature_2)
    return tf.reduce_sum(l1_diff, axis=axis)

def d_logistic(real_images, labels, generator, discriminator):
    # forward pass
    fake_images = generator(labels, training=True)

    real_scores, real_aux = discriminator(real_images, training=True)
    fake_scores, fake_aux = discriminator(fake_images, training=True)

    # gan loss
    d_gan_loss = tf.math.softplus(fake_scores)
    d_gan_loss += tf.math.softplus(-real_scores)

    # idt loss -> label 비교
    idt_loss = 0.5 * l1_loss(labels, real_aux, axis=[1])
    idt_loss += 0.5 * l1_loss(labels, fake_aux, axis=[1])

    return d_gan_loss, idt_loss


def d_logistic_r1_reg(real_images, labels, generator, discriminator):
    # forward pass
    fake_images = generator(labels, training=True)

    real_scores, real_aux = discriminator(real_images, training=True)
    fake_scores, fake_aux = discriminator(fake_images, training=True)

    # gan loss
    d_gan_loss = tf.math.softplus(fake_scores)
    d_gan_loss += tf.math.softplus(-real_scores)

    # idt loss
    idt_loss = 0.5 * l1_loss(labels, real_aux, axis=[1])
    idt_loss += 0.5 * l1_loss(labels, fake_aux, axis=[1])

    # gradient penalty
    with tf.GradientTape() as r1_tape:
        r1_tape.watch([real_images, labels])
        real_scores, real_aux = discriminator(real_images, training=True)
        real_loss = tf.reduce_sum(real_scores)

    real_grads = r1_tape.gradient(real_loss, real_images)
    r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
    r1_penalty = tf.expand_dims(r1_penalty, axis=1)

    return d_gan_loss, idt_loss, r1_penalty


def g_logistic_non_saturating(real_images, labels, generator, discriminator):
    # forward pass
    fake_images = generator(labels, training=True)
    fake_scores, fake_aux = discriminator(fake_images, training=True)
    # l1 pixel loss 
    pixel_loss = l1_loss(real_images, fake_images, axis=[1, 2, 3])

    # gan loss
    g_gan_loss = tf.math.softplus(-fake_scores)

    return g_gan_loss, pixel_loss


def g_logistic_ns_pathreg(real_images, labels, generator, discriminator,
                          pl_mean, pl_minibatch_shrink, pl_denorm, pl_decay):
    batch_size = tf.shape(real_images)[0]
    pl_minibatch = tf.maximum(1, tf.math.floordiv(batch_size, pl_minibatch_shrink))
    pl_labels = labels[:pl_minibatch]

    # forward pass
    fake_images, w_broadcasted = generator(labels, ret_w_broadcasted=True, training=True)
    fake_scores, fake_aux = discriminator(fake_images, training=True)
    g_gan_loss = tf.math.softplus(-fake_scores)

  
    # l1 pixel loss
    pixel_loss = l1_loss(real_images, fake_images, axis=[1, 2, 3])

    # Evaluate the regularization term using a smaller minibatch to conserve memory.
    with tf.GradientTape() as pl_tape:
        pl_tape.watch(pl_labels)
        pl_fake_images, pl_w_broadcasted = generator(pl_labels, ret_w_broadcasted=True, training=True)

        pl_noise = tf.random.normal(tf.shape(pl_fake_images)) * pl_denorm
        pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

    pl_grads = pl_tape.gradient(pl_noise_applied, pl_w_broadcasted)
    pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

    # Track exponential moving average of |J*y|.
    pl_mean_val = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
    pl_mean.assign(pl_mean_val)

    # Calculate (|J*y|-a)^2.
    pl_penalty = tf.square(pl_lengths - pl_mean)

    return g_gan_loss, pixel_loss, pl_penalty