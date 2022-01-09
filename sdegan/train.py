########################################
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
########################################

import argparse
from sdegan.sde_gan import Generator, Discriminator
from sdegan.utils import get_data, evaluate_loss
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm import tqdm
import os
import glob

def main(args):

    # Data
    data_info, dataloaders = get_data(args.batch_size, args.t_size, args.val_included)
    
    ts = data_info['ts']; data_size = data_info['data_size']
    train_dataloader = dataloaders['train']
    if args.val_included:
        val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']

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
    g_ini_init.scale = args.init_mult1
    g_func_init = keras.initializers.HeUniform()
    g_func_init.scale = args.init_mult2
    
    generator = Generator(data_size, args.init_noise_size, args.noise_size, 
                          args.hidden_size, args.mlp_size, args.num_layers, 
                          g_ini_init, g_func_init)

    discriminator = Discriminator(data_size, args.hidden_size, args.mlp_size, args.num_layers)


    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimizer = keras.optimizers.Adadelta(learning_rate=args.generator_lr)
    discriminator_optimizer = keras.optimizers.Adadelta(learning_rate=args.discriminator_lr)




    # Train both generator and discriminator.
    args.exp_name = f'SdeGan_epochs_{args.epochs}_ts_{args.t_size}_batch_{args.batch_size}_glr_{args.generator_lr}_dlr_{args.discriminator_lr}_wd_{args.weight_decay}_h_{args.hidden_size}_initN_{args.init_noise_size}_swa_{args.swa_step_start}_val_{args.val_included}'
    tot_batches = tf.data.experimental.cardinality(train_dataloader).numpy()
    print(args.exp_name)

    # Weight averaging really helps with GAN training.
    average_period = int((args.epochs - args.swa_step_start) * tot_batches)
    start_period = int(args.swa_step_start * tot_batches)
    averaged_generator = tfa.optimizers.SWA(generator_optimizer, start_averaging = start_period, average_period= average_period)
    averaged_discriminator = tfa.optimizers.SWA(discriminator_optimizer, start_averaging = start_period, average_period = average_period)


    args.save_path = os.path.join(args.save_path, args.exp_name)
    if (os.path.exists(args.save_path) == False):
        os.makedirs(args.save_path)

    args.model_path = os.path.join(args.model_path, args.exp_name)
    if (os.path.exists(args.model_path) == False):
        os.makedirs(args.model_path)

    train_log_dir = args.save_path + '/train'
    val_log_dir = args.save_path + '/val'
    test_log_dir = args.save_path + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        print(f'epoch: {epoch}')
        
        train_loss = tf.keras.metrics.Mean('train_loss', dtype = tf.float32)
        
        pbar = tqdm(total = tot_batches)
        for real_samples in train_dataloader:
        
            with tf.GradientTape() as tape:

                generated_samples = generator([ts, args.batch_size])
                generated_score = discriminator(generated_samples, generated=True)
                real_score = discriminator(real_samples, generated=False)
                loss = generated_score - real_score
                train_loss(loss)
                # Weight Decay (L2 regularization)
                gl2_loss =  tf.add_n([tf.nn.l2_loss(v) for v in generator.trainable_variables])
                dl2_loss =  tf.add_n([tf.nn.l2_loss(v) for v in discriminator.trainable_variables])
                loss += (gl2_loss + dl2_loss) * args.weight_decay
               
            g_gradients, d_gradients = tape.gradient(loss, [generator.trainable_variables, discriminator.trainable_variables])
            
            # maximize the generator's loss
            g_gradients *= -1
            
            averaged_generator.apply_gradients(zip(g_gradients, generator.trainable_variables))
            averaged_discriminator.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
            
            
            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            for layer_weight in discriminator.trainable_variables:
                if 'dense' and 'kernel' in layer_weight.name:
                    lim = 1. / layer_weight.shape[1] 
                    layer_weight.assign(tf.clip_by_value(layer_weight, -lim, lim))

            pbar.update(1)


        # after each epoch, evaluate model result on the validation data
        if args.val_included:
            val_loss = evaluate_loss(ts, val_dataloader, generator, discriminator, args.batch_size)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step = epoch)
                print(f'val_loss: {val_loss}')
            
            if abs(val_loss) < best_val_loss:
                best_val_loss = abs(val_loss)
                # save best val loss model
                for f in glob.glob(args.model_path + '/Best_*'):
                    os.remove(f)
                Gsnapshot_path = args.model_path + '/Best_G_snapshot_{}_'.format(args.exp_name) + 'loss_{:.4f}_epoch_{}.tf'.format(val_loss, epoch)
                generator.save_weights(Gsnapshot_path)
                Dsnapshot_path = args.model_path + '/Best_D_snapshot_{}_'.format(args.exp_name) + 'loss_{:.4f}_epoch_{}.tf'.format(val_loss, epoch)
                discriminator.save_weights(Dsnapshot_path)
                #import pdb; pdb.set_trace()

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step = epoch)
            print(f'train_loss: {train_loss.result()}')
        
        # save model
        for f in glob.glob(args.model_path + '/G_snapshot_*'):
            os.remove(f)
        for f in glob.glob(args.model_path + '/D_snapshot_*'):
            os.remove(f)

        Gsnapshot_path = args.model_path + '/G_snapshot_{}_'.format(args.exp_name) + 'loss_{:.4f}_epoch_{}.tf'.format(train_loss.result(), epoch)
        generator.save_weights(Gsnapshot_path)
        Dsnapshot_path = args.model_path + '/D_snapshot_{}_'.format(args.exp_name) + 'loss_{:.4f}_epoch_{}.tf'.format(train_loss.result(), epoch)
        discriminator.save_weights(Dsnapshot_path)

        #import pdb; pdb.set_trace()
        
        # reset
        train_loss.reset_states()        
        pbar.close()

    test_loss = evaluate_loss(ts, test_dataloader, generator, discriminator, args.batch_size)
    print(f'test_loss: {test_loss}')
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss, step = epoch)






def get_args():
    parser = argparse.ArgumentParser()
    # Data processing
    parser.add_argument('--t_size', nargs = '?', default = 22, type = int, help = 'How many days in one sample')
    parser.add_argument('--batch_size', nargs = '?', default = 32,  type = int, help =  'Batch size.')
    parser.add_argument('--val_included', dest = 'val_included', action = 'store_true', help = 'whether to include validation set')
    parser.add_argument('--val_not_included', dest = 'val_included', action = 'store_false', help = 'whether to include validation set')
    parser.set_defaults(val_included=True)

    # Architectural hyperparameters. These are quite small for illustrative purposes.
    parser.add_argument('--init_noise_size', nargs = '?', default = 3, type = int, help = 'How many noise dimensions to sample at the start of the SDE.')
    parser.add_argument('--noise_size', nargs = '?', default = 2, type = int, help = 'How many dimensions the Brownian motion has.')
    parser.add_argument('--hidden_size', nargs = '?', default = 8, type = int, help='How big the hidden size of the generator SDE and the discriminator CDE are.')
    parser.add_argument('--mlp_size', nargs = '?', default = 8, type = int, help = 'How big the layers in the various MLPs are.')
    parser.add_argument('--num_layers', nargs = '?', default = 1, type = int, help = 'How many hidden layers to have in the various MLPs.')

    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    parser.add_argument('--generator_lr', nargs = '?', default = 2e-3, type = float, help ='Learning rate often needs careful tuning to the problem.')
    parser.add_argument('--discriminator_lr', nargs = '?', default = 1e-3, type = float, help = 'Learning rate often needs careful tuning to the problem.')
    parser.add_argument('--epochs', nargs = '?', default = 200,  type = int, help = 'How many steps to train both generator and discriminator for.')
    parser.add_argument('--init_mult1', nargs = '?', default = 1, type = int, help = 'Changing the initial parameter size can help.')
    parser.add_argument('--init_mult2', nargs = '?', default = 1, type = float )         
    parser.add_argument('--weight_decay', nargs = '?', default = 0.01, type = float, help = 'Weight decay.')
    parser.add_argument('--swa_step_start', nargs = '?', default = 180, type = int, help = 'When to start using stochastic weight averaging.')
    
    # Evaluate and save
    parser.add_argument('--save_path', nargs = '?', default = 'logs', type = str, help = 'log path')
    parser.add_argument('--model_path', nargs = '?', default = 'models', type = str, help = 'model path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
    
