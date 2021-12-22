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

"""Train an SDE as a GAN, on data from a time-dependent Ornstein--Uhlenbeck process.

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".
https://arxiv.org/abs/2102.03657

This reproduces the toy example in Section 4.1 of that paper.

This additionally uses the improvements introduced in "Efficient and Accurate Gradients for Neural SDEs".
https://arxiv.org/abs/2105.13493

To run this file, first run the following to install extra requirements:
pip install fire
pip install git+https://github.com/patrick-kidger/torchcde.git

To run, execute:
python -m examples.sde_gan
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tfcde
import tfsde
import tqdm


###################
# First some standard helper objects.
###################

# Set the default initializer as HeUniform
default_initializer = keras.initializers.HeUniform()

class LipSwish(layers.Layer):
    def call(self, x):
        return 0.909 * tf.nn.silu(x)


class MLP(layers.Layer):
    def __init__(self, out_size, mlp_size, num_layers, tanh, initializer=None):
        super().__init__()
        
        if initializer is None:
            initializer = default_initializer

        model_layers = [layers.Dense(mlp_size, kernel_initializer = initializer, bias_initializer=initializer),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model_layers.append(layers.Dense(mlp_size, kernel_initializer = initializer, bias_initializer=initializer))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model_layers.append(LipSwish())
        model_layers.append(layers.Dense(out_size, kernel_initializer = initializer, bias_initializer=initializer))

        if tanh:
            model_layers.append(layers.Activation('tanh'))
        
        self.model_layers = model_layers

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x



###################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
###################
class GeneratorFunc(layers.Layer):
    sde_type = 'stratonovich'
    noise_type = 'general'

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
        self._drift = MLP(hidden_size, mlp_size, num_layers, tanh=True, initializer = initializer)
        self._diffusion = MLP(hidden_size * noise_size, mlp_size, num_layers, tanh=True, initializer = initializer)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = tf.expand_dims(tf.repeat(t, [x.shape[0]]), -1)
        tx = tf.concat([t, x], axis=1)
        return self._drift(tx), tf.reshape(self._diffusion(tx), [x.shape[0], self._hidden_size, self._noise_size])


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(keras.Model):
    def __init__(self, data_size, initial_noise_size, noise_size, 
                hidden_size, mlp_size, num_layers, g_ini_init=None, g_func_init=None):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP( hidden_size, mlp_size, num_layers, tanh=False, initializer = g_ini_init)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers, initializer = g_func_init)

        self._readout = layers.Dense(data_size, kernel_initializer = default_initializer, bias_initializer= default_initializer)

    def call(self, inputs):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.
        ts, batch_size = inputs

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = tf.random.normal([batch_size, self._initial_noise_size])
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = tfsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = tf.transpose(xs, [1, 0, 2])
        ys = self._readout(xs)
        

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = tf.tile(tf.expand_dims(tf.expand_dims(ts, -1), 0), [batch_size,1,1])
        return tfcde.linear_interpolation_coeffs(tf.concat([ts, ys], axis=2))


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################
class DiscriminatorFunc(layers.Layer):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        # input_size = 1 + hidden_size
        self._module = MLP(hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

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

        self._initial = MLP( hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = layers.Dense(1, kernel_initializer = default_initializer, bias_initializer= default_initializer)

    def call(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = tfcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = tfcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) )
        score = self._readout(hs[:, -1])
        return tf.reduce_mean(score)


##################
# Fix CDE
##################

def get_data_np(batch_size=1024):
    t_size = 64

    with open('../torchsde/ys_data.npy', 'rb') as f:
        ys = np.load(f)

    ys = tf.constant(ys)

    ts = tf.linspace(0, t_size - 1, t_size)
    # NOTE: ts is of type float64, cast it to float32 to be cautious
    ts = tf.cast(ts, dtype=tf.float32)
    
    data_size = ys.shape[-1] - 1  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = np.load('tf_data.npy', 'r')
     
    dataloader = tf.data.Dataset.from_tensor_slices(ys_coeffs)
    dataloader = dataloader.shuffle(ys.shape[0], reshuffle_each_iteration = True)
    dataloader = dataloader.batch(batch_size)
    #dataset = torch.utils.data.TensorDataset(ys_coeffs)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ts, data_size, dataloader



###################
# Now do normal GAN training, and plot the results.
#
# GANs are famously tricky and SDEs trained as GANs are no exception. Hopefully you can learn from our experience and
# get these working faster than we did -- we found that several tricks were often helpful to get this working in a
# reasonable fashion:
# - Stochastic weight averaging (average out the oscillations in GAN training).
# - Weight decay (reduce the oscillations in GAN training).
# - Final tanh nonlinearities in the architectures of the vector fields, as above. (To avoid the model blowing up.)
# - Adadelta (interestingly seems to be a lot better than either SGD or Adam).
# - Choosing a good learning rate (always important).
# - Scaling the weights at initialisation to be roughly the right size (chosen through empirical trial-and-error).
###################

def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    total_samples = 0
    total_loss = 0
    for real_samples, in dataloader:
        generated_samples = generator.evaluate(ts, batch_size)
        generated_score = discriminator.evaluate(generated_samples)
        real_score = discriminator.evaluate(real_samples)
        loss = generated_score - real_score
        total_samples += batch_size
        total_loss += loss * batch_size
    return total_loss.numpy() / total_samples


def main(
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        initial_noise_size=5,  # How many noise dimensions to sample at the start of the SDE.
        noise_size=3,          # How many dimensions the Brownian motion has.
        hidden_size=16,        # How big the hidden size of the generator SDE and the discriminator CDE are.
        mlp_size=16,           # How big the layers in the various MLPs are.
        num_layers=1,          # How many hidden layers to have in the various MLPs.

        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=2e-4,      # Learning rate often needs careful tuning to the problem.
        discriminator_lr=1e-3,  # Learning rate often needs careful tuning to the problem.
        batch_size=1024,        # Batch size.
        steps=10000,            # How many steps to train both generator and discriminator for.
        init_mult1=2,           # Changing the initial parameter size can help.
        init_mult2=0.5,         #
        weight_decay=0.01,      # Weight decay.
        swa_step_start=5000,    # When to start using stochastic weight averaging.

        # Evaluation and plotting hyperparameters
        steps_per_print=10,                   # How often to print the loss.
        num_plot_samples=50,                  # How many samples to use on the plots at the end.
):

    # Data
    #ts, data_size, train_dataloader = get_data(batch_size=batch_size, device=device)
    ts, data_size, train_dataloader = get_data_np(batch_size=batch_size)

    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models

    # Modifying Weights Initialization
    # init_mult1 and init_mult2 are used for weights initilization. In Pytorch, 
    # Kaiming_uniform is used for linear layers, which has the formula U[-sqrt(1/in_features), -sqrt(1/in_features)]
    # In tf, the distribution derived from the same paper is referred to as He_uniform and is defined as 
    # U[-sqrt(6/in_features), sqrt(6/in_features)]. The default initializer is Xavier uniform initializer,
    # which is defined as U[-sqrt(6/(in_features+out_features), sqrt(6/in_features+out_features)]
    # Due to the difference, we add initilizer parameter for MLP layer for initializer adjustment

    # Initialize He_uniform.     
    g_ini_init = keras.initializers.HeUniform()
    # 6 in the formula is computed as scale * 3. changing the scale value
    # to change the scale of the initialized parameters
    # Default scale value is 2
    g_ini_init.scale = init_mult1
    g_func_init = keras.initializers.HeUniform()
    g_func_init.scale = init_mult2
    
    # Weight Decay (L2 regularization)
    # Instead of introducing weight decay in optimizers in torch, tf introduces regularizers in each layer
    # or by explictly adding the penalty to the loss 
    # TODO: Weight Decay is disabled now, add back later 
    generator = Generator(data_size, initial_noise_size, noise_size, 
                          hidden_size, mlp_size, num_layers, 
                          g_ini_init, g_func_init)

    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers)


    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimizer = keras.optimizers.Adadelta(learning_rate=generator_lr)
    discriminator_optimizer = keras.optimizers.Adadelta(learning_rate=discriminator_lr)


    # Weight averaging really helps with GAN training.
    average_period = steps - swa_step_start
    averaged_generator = tfa.optimizers.SWA(generator_optimizer, start_averaging=swa_step_start, average_period=average_period)
    averaged_discriminator = tfa.optimizers.SWA(discriminator_optimizer, start_averaging=swa_step_start, average_period=average_period)


    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        #real_samples = next(infinite_train_dataloader)
        
        #########################
        # DEBUG
        #########################
        # use the same input batch for debugging
        batch_size = 2
        real_samples = np.load('../torchsde/debug_batch.npy')
        real_samples = tf.constant(real_samples)
        ########################
        
        with tf.GradientTape() as tape:

            generated_samples = generator([ts, batch_size])
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            
            g_gradient = tape.gradient(loss, generator.trainable_variables)
            d_gradient = tape.gradient(loss, discriminator.trainable_variables)
            
            import pdb; pdb.set_trace()

        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
            if step > swa_step_start:
                total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
                                                    averaged_discriminator.module)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    _, _, test_dataloader = get_data(batch_size=batch_size, device=device)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()
    

