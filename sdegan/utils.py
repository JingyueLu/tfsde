import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random
import tfsde


#############################
# Data preprocessing
##############################

def get_data(batch_size, val_included):
    '''
    We assume the underlying model for the data is 
        
        dS/S = a dt + b dW

    '''
    # open downloaded csv file
    hsi = pd.read_csv('sdegan/Daily_HSI.csv').dropna()

    # we use the adjust close values 
    hsi_close_adj = hsi["Adj Close"].values

    # since we assume the underlying model follows
    # geometric brownian motion, we model the log values with sdegan
    hsi_close_adj = np.log(hsi_close_adj)
    data_info = {}
    diff = hsi_close_adj[1:] - hsi_close_adj[:len(hsi_close_adj)-1]
    diff_mean = tf.math.reduce_mean(diff).numpy()
    diff_std = tf.math.reduce_std(diff).numpy()
    print(f'\n HSI Data: mean {diff_mean} std {diff_std}\n')
    data_info = {}
    data_info['data_mean'] = diff_mean
    data_info['data_std'] = diff_std
    
    # split the data
    t_size = 2
    ts = tf.linspace(0, t_size - 1, t_size)
    # NOTE: ts is of type float64, cast it to float32 to be cautious
    ts = tf.cast(ts, dtype=tf.float32)

    dataset = [tf.stack([ts, hsi_close_adj[i:i+t_size]-hsi_close_adj[i]], axis=1) for i in range(0, len(hsi_close_adj)-1)]
    random.shuffle(dataset)
    if val_included:
        train_num = int(len(dataset)*0.6)
        val_num = int(len(dataset)*0.8)
        train = dataset[:train_num]
        val = dataset[train_num:val_num]
        test = dataset[val_num:]

    else:
        train_num = int(len(dataset)*0.7)
        train = dataset[:train_num]
        test = dataset[train_num:]
    
    data_size = 1  # How many channels the data has (not including time).

    # Build Data Loaders

    dataloaders = {}
    dataloaders['train'] = tf.data.Dataset.from_tensor_slices(train).shuffle(len(train),reshuffle_each_iteration = True).batch(batch_size)
    dataloaders['test'] = tf.data.Dataset.from_tensor_slices(test).shuffle(len(test),reshuffle_each_iteration = True).batch(batch_size)

    if val_included:
        dataloaders['val'] = tf.data.Dataset.from_tensor_slices(val).shuffle(len(val),reshuffle_each_iteration = True).batch(batch_size)
    
    data_info['ts'] = ts; data_info['data_size'] = data_size
    return data_info, dataloaders






#############################
# Training Evaluation
#############################

def evaluate_loss(ts, dataloader, generator, discriminator, batch_size):
    
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype = tf.float32)
    for real_samples in dataloader:
        generated_samples = generator([ts, batch_size])
        generated_score = discriminator(generated_samples, generated=False)
        real_score = discriminator(real_samples, generated=False)
        loss = generated_score - real_score
        eval_loss(loss)

    return eval_loss.result()






#############################
# Synthetic Data
#############################
'''
    we generate synthetic data from the process dS/S = 0.1dt + 0.2dW
'''

def get_synthetic_data(batch_size, val_included):

    dataset_size = 2000
    t_size = 2

    class SimpleSDE(keras.Model):
        sde_type = 'stratonovich'
        noise_type = 'additive'

        def __init__(self, mu, sigma):
            super().__init__()
            self.mu = tf.constant([[mu]], dtype=tf.float32)
            self.sigma = tf.constant([[sigma]], dtype=tf.float32)

        def f(self, t, y):
            return tf.tile(self.mu, y.shape)

        def g(self, t, y):
            return tf.expand_dims(tf.tile(self.sigma, y.shape), axis=2)

    sde = SimpleSDE(mu=0.08, sigma=0.2)
    y0 = tf.random.uniform([dataset_size, 1]) 
    ts = tf.linspace(0, t_size - 1, t_size)
    ts = tf.cast(ts, dtype=tf.float32)
    ys = tfsde.sdeint(sde, y0, ts, method='reversible_heun')
    # use Variable to assign new values
    ys = tf.Variable(ys)
    ys[1].assign(ys[1] - ys[0])
    ys[0].assign(ys[0]*0)
    ys = tf.constant(ys)
    diff = ys[1] - ys[0]
    diff_mean = tf.math.reduce_mean(diff).numpy()
    diff_std = tf.math.reduce_std(diff).numpy()
    print(f'\n Synthetic Data: mean {diff_mean} std {diff_std}\n')
    data_info = {}
    data_info['data_mean'] = diff_mean
    data_info['data_std'] = diff_std



    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(ts, axis=1), axis=0), [dataset_size, 1, 1]),
                    tf.transpose(ys, perm=[1, 0, 2])], axis=2)
    # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

    ###################
    # Package up.
    ###################
    data_size = ys.shape[-1] - 1  # How many channels the data has (not including time, hence the minus one).
    dataloaders = {}
    if val_included:
        train_num = int(len(ys)*0.6)
        val_num = int(len(ys)*0.8)
        ystrain = ys[:train_num]
        ysval = ys[train_num:val_num]
        ystest = ys[val_num:]
        dataloaders['val'] = tf.data.Dataset.from_tensor_slices(ysval).shuffle(len(ysval),reshuffle_each_iteration = True).batch(batch_size)

    else:
        train_num = int(len(ys)*0.7)
        ystrain = ys[:train_num]
        ystest = ys[train_num:]


    dataloaders['train'] = tf.data.Dataset.from_tensor_slices(ystrain).shuffle(len(ystrain),reshuffle_each_iteration = True).batch(batch_size)
    dataloaders['test'] = tf.data.Dataset.from_tensor_slices(ystest).shuffle(len(ystest),reshuffle_each_iteration = True).batch(batch_size)

    data_info['ts'] = ts; data_info['data_size'] = data_size
    return data_info, dataloaders












