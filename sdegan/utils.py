import tensorflow as tf
import pandas as pd
import numpy as np


#############################
# Data preprocessing
##############################

def get_data(batch_size, t_size, val_included=False):
    '''
    Input: 
        val_included: Boolean whether to include the validation set
                      when included, we use train(70%), val(15%), test(15%)
                      when not included, we use train(80%) test(20%)
    '''
    # open downloaded csv file
    hsi = pd.read_csv('sdegan/Daily_HSI.csv').dropna()

    # we use the adjust close values 
    hsi_close_adj = hsi["Adj Close"].values

    # since we assume the underlying model follows
    # geometric brownian motion, we model the log values with sdegan
    hsi_close_adj = np.log(hsi_close_adj)
    
    # split the data
    ts = tf.linspace(0, t_size - 1, t_size)
    # NOTE: ts is of type float64, cast it to float32 to be cautious
    ts = tf.cast(ts, dtype=tf.float32)

    def normalise_split(rtrain, rval, rtest, hsi_close_adj, t_size, ts):
        if rval == 0:
            total_num_data = len(hsi_close_adj) - 2*(t_size-1)
        else: 
            total_num_data = len(hsi_close_adj) - 3*(t_size-1)

        train_num = round(total_num_data * rtrain)
        val_num = round(total_num_data * rval)
        test_num = total_num_data - train_num - val_num
        
        train = hsi_close_adj[0:train_num+t_size-1]
        train_mean = train.mean()
        train_std = train.std()
        
        # normalise the data
        hsi_close_adj = (hsi_close_adj - train_mean) / train_std
        

        # split the data
        if rval == 0:
            train = [tf.stack([ts, hsi_close_adj[i:i+t_size]], axis=1) for i in range(0, train_num)]
            test = [tf.stack([ts, hsi_close_adj[i:i+t_size]], axis=1) for i in range(train_num + t_size-1, len(hsi_close_adj) - t_size + 1)]
            return train, test, train_mean, train_std
        else:
            train = [tf.stack([ts, hsi_close_adj[i:i+t_size]], axis=1) for i in range(0, train_num)]
            val = [tf.stack([ts, hsi_close_adj[i:i+t_size]], axis=1) for i in range(train_num + t_size-1, train_num + t_size-1 + val_num )]
            test = [tf.stack([ts, hsi_close_adj[i:i+t_size]], axis=1) for i in range(train_num + val_num + 2*t_size-2, len(hsi_close_adj) - t_size+1)]
            return train, val, test, train_mean, train_std

    if val_included:
        train, val, test, train_mean, train_std = normalise_split(0.7, 0.15, 0.15, hsi_close_adj, t_size, ts)
    else:
        train, test, train_mean, train_std = normalise_split(0.8, 0., 0.2, hsi_close_adj, t_size, ts)

    
    data_size = 1  # How many channels the data has (not including time).

    # Build Data Loaders

    dataloaders = {}
    dataloaders['train'] = tf.data.Dataset.from_tensor_slices(train).shuffle(len(train),reshuffle_each_iteration = True).batch(batch_size)
    dataloaders['test'] = tf.data.Dataset.from_tensor_slices(test).shuffle(len(test),reshuffle_each_iteration = True).batch(batch_size)

    
    if val_included:
        dataloaders['val'] = tf.data.Dataset.from_tensor_slices(val).shuffle(len(val),reshuffle_each_iteration = True).batch(batch_size)

    
    data_info = {}
    data_info['ts'] = ts; data_info['data_size'] = data_size
    data_info['train_mean'] = train_mean; data_info["train_std"] = train_std
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





