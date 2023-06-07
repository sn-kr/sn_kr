from tensorflow import keras

import numpy as np
import tensorflow as tf


class SReFT(keras.Model):

    def __init__(self,
                 output_dim=4,
                 latent_dim=32,
                 activation='tanh',
                 random_state=None):

        super(SReFT, self).__init__()

        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.activation = activation
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.tracker_tr = keras.metrics.Mean()
        self.tracker_va = keras.metrics.Mean()

        self.lnvar_y = tf.Variable(tf.zeros(self.output_dim))

        self.model_1 = keras.Sequential([
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(1, activation=tf.nn.relu),])
            # keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),])

        self.model_y = keras.Sequential([
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(self.latent_dim, activation=self.activation),
            keras.layers.Dense(self.output_dim, activation=None),])

        return None

    def call(self, inputs, training=False):
        (input1, input2) = inputs
        offset = self.model_1(input1, training=training)
        input2 = tf.concat((input2[:, :, :1] + offset, input2[:, :, 1:]), axis=-1)
        y_pred = self.model_y(input2, training=training)
        return y_pred

    def train_step(self, batch):
        inputs, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            y_loss = self.compute_negative_log_likelihood(y_true, y_pred)
            objval = tf.reduce_sum(y_loss)
        grads = tape.gradient(objval, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.tracker_tr.update_state(y_loss)
        return {'loss': self.tracker_tr.result(),}

    def test_step(self, batch):
        inputs, y_true = batch
        y_pred = self(inputs, training=False)
        y_loss = self.compute_negative_log_likelihood(y_true, y_pred)
        self.tracker_va.update_state(y_loss)
        return {'loss': self.tracker_va.result(),}

    def compute_negative_log_likelihood(self, y_true, y_pred):
        is_nan = tf.math.is_nan(y_true)
        y_true = tf.where(is_nan, tf.zeros_like(y_true), y_true)
        y_pred = tf.where(is_nan, tf.zeros_like(y_pred), y_pred)
        neg_ll = self.lnvar_y + tf.pow(y_true - y_pred, 2) / tf.exp(self.lnvar_y)
        neg_ll = tf.where(is_nan, tf.zeros_like(neg_ll), neg_ll)
        return tf.reduce_sum(neg_ll, axis=(1, 2))



class DummyTransformer():

    def __init__(self,):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X
