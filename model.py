import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Input, Reshape, TimeDistributed, RepeatVector, Conv1DTranspose, GRU, Dropout, Concatenate, BatchNormalization
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from keras.constraints import Constraint
import keras.backend as K

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

from objectives import compute_loss

from utils import *

print(tf.__version__)


@dataclass
class Config(object):
    code_size: int = 40
    seq_len: int = 500
    #batch_size: int = 16
    num_channels: int = 1
    fmap_base: int = 8192
    fmap_decay: int = 1.0
    fmap_max: int = 512
    mapper_layers: int = 4
    seed: int = 42
    # position: str ='Torso'
    # channel: str ='Acc_x'
    batch_size: int = 32
    epochs: int = 512
    learning_rate: int = 0.0001

    def __str__(self):
        res = 'VaeConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class AbstractionVae(keras.Model):

    def __init__(self, config: Config, optimizer, seed, **kwargs):
        super(AbstractionVae, self).__init__(**kwargs)
        self.config = config
        self.optimizer = optimizer
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        tf.random.set_seed(config.seed)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #self.compile(optimizer=keras.optimizers.RMSprop(learning_rate=config.learning_rate))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, channel: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        return self.encoder(channel)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        return self.decoder(z)

    def build_encoder(self) -> tf.keras.Model:
        ts = Input(shape=(500,))  # , name=self.config.position+'_'+self.config.channel)
        #print(tf.shape(ts))
        x = Reshape((500, 1))(ts)

        x = Conv1D(filters=64, kernel_size=5, strides=2, activation='relu',
                   kernel_constraint=Between(-0.08, 0.08),
                   bias_constraint=Between(-0.08, 0.08))(x)
        # Between(): constrain the weights of the layers to circumvent nan during training. Suggested here https://github.com/y0ast/VAE-Torch/issues/3
        #x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv1D(filters=64, kernel_size=5, strides=2, activation='relu',
                   kernel_constraint=Between(-0.08, 0.08),
                   bias_constraint=Between(-0.08, 0.08))(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(32, activation='relu', return_sequences=True,
                 kernel_constraint=Between(-0.08, 0.08),
                 bias_constraint=Between(-0.08, 0.08))(x)
        x = LSTM(self.config.code_size + self.config.code_size, activation=None,
                 kernel_constraint=Between(-0.08, 0.08),
                 bias_constraint=Between(-0.08, 0.08))(x)

        #z_mean = Dense(self.config.code_size)(x)
        #z_log_std = Dense(self.config.code_size)(x)
        z_mean, z_log_std = tf.split(x, num_or_size_splits=2, axis=1)
        z = Sampling()([z_mean, z_log_std])

        #print(tf.shape(z))
        encoder = tf.keras.Model(inputs=ts, outputs=[z_mean, z_log_std, z])
        encoder.summary()
        return encoder

    def build_decoder(self) -> tf.keras.Model:
        z = Input(shape=(self.config.code_size,))
        #x = RepeatVector(500)(z)
        x = Dense(125*1, kernel_constraint=Between(-0.08, 0.08),
                  bias_constraint=Between(-0.08, 0.08))(z)
        x = Reshape((125, 1))(x)
        x = Conv1DTranspose(filters=64, kernel_size=7, strides=2,
                            activation='relu', padding='same',
                            kernel_constraint=Between(-0.08, 0.08),
                            bias_constraint=Between(-0.08, 0.08))(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv1DTranspose(filters=64, kernel_size=7, strides=2,
                            activation='relu', padding='same',
                            kernel_constraint=Between(-0.08, 0.08),
                            bias_constraint=Between(-0.08, 0.08))(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(32, activation='relu', return_sequences=True,
                 name='decoder_lstm_0', kernel_constraint=Between(-0.08, 0.08),
                 bias_constraint=Between(-0.08, 0.08))(x)
        #x = LSTM(1, activation=None, return_sequences=True, name='decoder_lstm_1')(x)
        x = TimeDistributed(keras.layers.Dense(1))(x)
        x = Reshape((500,))(x)

        decoder = tf.keras.Model(inputs=z, outputs=x)
        decoder.summary()
        return decoder

    #@tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(self, x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return (
            self.total_loss_tracker.result(),  # "loss"
            self.reconstruction_loss_tracker.result(),  # "reconstruction_loss"
            self.kl_loss_tracker.result(),  # "kl_loss"
        )

    def set_params(self, model_params=None):
        if model_params is not None:
            for param, new_param in zip(self.get_params(), model_params):
                param.assign(new_param)

    def set_gradientParam(self, preG, preGn):
        self.optimizer.set_preG(preG, self)
        self.optimizer.set_preGn(preGn, self)

    def set_vzero(self, vzero):
        pass

    def get_params(self):
        return self.trainable_variables

    def get_gradients(self, train_data, model_len):
        """
        Compute gradients w.r.t. (a part of) `train_data` without applying them.
        """
        num_samples = 100
        x = train_data['x']['Acc_m']  # TODO
        idx = np.random.choice(np.arange(len(x)), num_samples, replace=False)
        x = x[idx]

        with tf.GradientTape() as tape:
            loss = compute_loss(self, x)
        gradients = tape.gradient(loss, self.trainable_variables)

        # flatten returned gradients
        flattenedList = [K.flatten(g) for g in gradients]
        gradients = K.concatenate(flattenedList)
        # print(gradients)

        return num_samples, gradients

    def get_raw_gradients(self, train_data):
        num_samples = 100
        x = train_data['x']['Acc_m']  # TODO
        idx = np.random.choice(np.arange(len(x)), num_samples, replace=False)
        x = x[idx]

        with tf.GradientTape() as tape:
            loss = compute_loss(self, x)
        gradients = tape.gradient(loss, self.trainable_variables)

        return gradients

    def solve_grad(self):
        pass

    def solve_inner(self, optimizer, data, num_epochs=1, batch_size=10):
        """
        Solves local optimization problem

        Returns
            1: soln: local optimization solution
            2: grad: computed gradients
            3: comp: TODO number of FLOPs executed in training process
        """

        def train(model, data):
            """ Trains model with the given data """
            n_batches = len(data) // batch_size

            def print_info(batch, epoch, recon_err, kl, loss):
                """ Print training info """
                str_out = " recon: {}".format(round(float(recon_err), 2))
                str_out += " kl: {}".format(round(float(kl),2))
                # str_out += " capacity (nats): {}".format(round(float(model.C), 2))
                progress_bar(batch, n_batches, loss, epoch, num_epochs, suffix=str_out)

            def get_batch(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield iterable[ndx:min(ndx + n, l)]

            # Training loop
            for epoch in range(num_epochs):
                for batch, X in enumerate(get_batch(data, batch_size)):
                    loss, recon_err, kl = model.train_step(X)

                    # print_info(batch, epoch, recon_err, kl, loss)

        # get model's parameters after training for `num_epochs` epochs
        num_samples = 100
        x = data['x']['Acc_m']  # TODO
        idx = np.random.choice(np.arange(len(x)), num_samples, replace=False)
        x = x[idx]
        train(self, x)
        soln = self.get_params()

        # get gradients with respect to ... ?
        _, grad = self.get_gradients(data, model_len=0)

        # TODO get number of FLOPS
        comp = 0

        return soln, grad, comp

    def test(self, eval_data):
        tot_correct = 0
        loss = 0.0
        return tot_correct, loss
