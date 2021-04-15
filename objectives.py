import tensorflow as tf
from tensorflow import keras
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def aux(cross_ent, z, mean, logvar):
    logpx_z = -tf.reduce_sum(cross_ent, axis=0)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return logpx_z, logpz, logqz_x


def compute_loss(model, x):
    mean, logvar, z = model.encode(x)
    x_logit = model.decode(z)
    cross_ent = keras.losses.mse(x, x_logit)
    logpx_z, logpz, logqz_x = aux(cross_ent, z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def beta_cycling_loss(model, x, beta):
    mean, logvar, z = model.encode(x)
    x_logit = model.decode(z)
    cross_ent = keras.losses.mse(x, x_logit)
    logpx_z, logpz, logqz_x = aux(cross_ent, z, mean, logvar)

    # Cyclical Annealing Schedule [1] Hao Fu et al. 2019 (NAACL)
    return -tf.reduce_mean(logpx_z + beta * (logpz - logqz_x))


def decomposition_loss(model, x, beta, alpha):
    mean, logvar, z = model.encode(x)
    x_logit = model.decode(z)
    cross_ent = keras.losses.mse(x, x_logit)
    logpx_z, logpz, logqz_x = aux(cross_ent, z, mean, logvar)

    # Objective enforcing decomposition [2] Mathieu, Emile, et al. 2019 (ICML)
    return -tf.reduce_mean(logpx_z + beta * (logpz - logqz_x))
