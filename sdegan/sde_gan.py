# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".
https://arxiv.org/abs/2102.03657

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tfcde
import tfsde


####################################
# First some standard helper objects.
####################################

# Set the default initializer as HeUniform
default_initializer = keras.initializers.HeUniform()
default_initializer.scale = 1


class AdditiveLayer(layers.Layer):
    def __init__(self, units=32):
        super(AdditiveLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.zeros_initializer()
        self.w = tf.Variable(
                initial_value = w_init(shape = (input_shape[-1], self.units),
                                       dtype = 'float32'),
                trainable = False
                )
        b_init = tf.random_uniform_initializer()
        self.b = tf.Variable(
                initial_value = b_init(shape=(self.units,), dtype='float32'),
                trainable = True
                )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



########################################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
#######################################

class GeneratorFunc(layers.Layer):
    sde_type = 'stratonovich'
    noise_type = 'additive'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, initializer=None):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        
        # NOTE: input_size = 1 + hidden_size
        #self._drift = AdditiveLayer(hidden_size)
        #self._diffusion = AdditiveLayer(hidden_size)
        self._drift = tf.Variable(tf.random.uniform([1]), name = 'drift', trainable=True)
        self._diffusion = tf.Variable(tf.random.uniform([1]), name = 'diffusion', trainable=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        #t = tf.expand_dims(tf.repeat(t, [x.shape[0]]), -1)
        #tx = tf.concat([t, x], axis=1)
        return x*0 + self._drift, tf.expand_dims(x*0 + self._diffusion, axis=2)


## Now we wrap it up into something that computes the SDE.

class Generator(keras.Model):
    def __init__(self, data_size, initial_noise_size, noise_size, 
                hidden_size, mlp_size, num_layers, g_ini_init=None, g_func_init=None):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        #self._initial = MLP( hidden_size, mlp_size, num_layers, tanh=False, initializer = g_ini_init)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers, initializer = g_func_init)

        #self._readout = layers.Dense(data_size, kernel_initializer = default_initializer, bias_initializer= default_initializer)

    def call(self, inputs):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        ts, batch_size = inputs

        # Actually solve the SDE.
        init_noise = tf.random.normal([batch_size, self._initial_noise_size])
        
        x0 = tf.zeros([batch_size, 1])

        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        xs = tfsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = tf.transpose(xs, [1, 0, 2])
        #ys = self._readout(xs)
        
        ## Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        
        ts = tf.tile(tf.expand_dims(tf.expand_dims(ts, -1), 0), [batch_size,1,1])

        return tf.concat([ts, xs], axis=2)


########################################
# Next the discriminator. 
# 
# Here, we're going to use a neural controlled differential equation 
# (neural CDE) as the discriminator, just as in the 
# "Neural SDEs as Infinite-Dimensional GANs" paper. 
# (You could use other things as well,but this is a natural choice.)
#
########################################

class DiscriminatorFunc(layers.Layer):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        # input_size = 1 + hidden_size
        self._module = layers.Dense(hidden_size * (1 + data_size))

    def call(self, inputs):
        t, h = inputs
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = tf.expand_dims(tf.repeat(t, [h.shape[0]]), -1)
        th = tf.concat([t, h], axis=1)
        return tf.reshape(self._module(th), [h.shape[0], self._hidden_size, 1 + self._data_size])


class Discriminator(keras.Model):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = layers.Dense(hidden_size)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = layers.Dense(1, kernel_initializer = default_initializer, bias_initializer= default_initializer)

    def call(self, ys_coeffs, generated):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = tfcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        if generated:
            hs = tfcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', dt=1.0,
                                 adjoint_method='adjoint_reversible_heun',
                                 adjoint_params=(ys_coeffs,) )
        else:
            hs = tfcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', dt=1.0,
                                 adjoint_method='adjoint_reversible_heun',
                                 adjoint_params=() )
        
        score = self._readout(hs[:, -1])
        return tf.reduce_mean(score)






