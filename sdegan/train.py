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
from sdegan.utils import get_data, evaluate_loss, get_synthetic_data
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm import tqdm
import os
import glob

def main(args):

    # Data
    if args.synthetic:
        data_info, dataloaders = get_synthetic_data(args.batch_size, args.val_included)
    else:
        data_info, dataloaders = get_data(args.batch_size, args.val_included)
    
    ts = data_info['ts']; data_size = data_info['data_size']
    train_dataloader = dataloaders['train']
    if args.val_included:
        val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']


    
    generator = Generator(data_size, args.init_noise_size, args.noise_size, 
                          args.hidden_size, args.mlp_size, args.num_layers)

    discriminator = Discriminator(data_size, args.hidden_size, args.mlp_size, args.num_layers)


    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimizer = keras.optimizers.Adadelta(learning_rate=args.generator_lr)
    discriminator_optimizer = keras.optimizers.Adadelta(learning_rate=args.discriminator_lr)

    # Use a log to record generator parameters with good performance
    log_name = f'./sdegan/SdeGan_epochs_{args.epochs}_batch_{args.batch_size}_val_{args.val_included}_Synthetic_{args.synthetic}.log'
    answ = os.path.exists(log_name)
    if answ:
        flog = open(log_name, 'a')
        flog.write('\n\n')

    else:
        flog = open(log_name, 'w')


    # Train both generator and discriminator.
    args.exp_name = f'SdeGan_epochs_{args.epochs}_ts_2_batch_{args.batch_size}_glr_{args.generator_lr}_dlr_{args.discriminator_lr}_wd_{args.weight_decay}_h_{args.hidden_size}_initN_{args.init_noise_size}_val_{args.val_included}_Synthetic_{args.synthetic}_exp{args.exp_no}'
    tot_batches = tf.data.experimental.cardinality(train_dataloader).numpy()
    print(args.exp_name)
    flog.write(args.exp_name)
    flog.write('\n')


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
               
            g_gradients, d_gradients = tape.gradient(loss, [generator.trainable_variables, discriminator.trainable_variables])
            
            # maximize the generator's loss
            neg_g_gradients  = [-1*i for i in g_gradients]

            generator_optimizer.apply_gradients(zip(neg_g_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
            
            
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
                print(f'Current Estimates: Drift {generator.trainable_variables[0].numpy()[0]} Diffusion {generator.trainable_variables[1].numpy()[0]}')
                best_val_drift = generator.trainable_variables[0].numpy()[0]
                best_val_diffusion = generator.trainable_variables[1].numpy()[0]
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step = epoch)
            print(f'train_loss: {train_loss.result()}')
            if abs(train_loss.result()) < 1e-3:
                print(f'Current Estimates: Drift {generator.trainable_variables[0].numpy()[0]} Diffusion {generator.trainable_variables[1].numpy()[0]}')
                flog.write(f'loss={train_loss.result()} Estimates: Drift {generator.trainable_variables[0].numpy()[0]} Diffusion {generator.trainable_variables[1].numpy()[0]}\n')
                flog.flush()
        
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
    data_mean = data_info['data_mean']
    data_std = data_info['data_std']
    print(f'Data Info: mean {data_mean} std {data_std}')
    print(f'Estimated(Final) Drift {generator.trainable_variables[0].numpy()[0]} Diffusion {generator.trainable_variables[1].numpy()[0]}')
    print(f'Estimated(BestVal) Drift {best_val_drift} Diffusion {best_val_diffusion}\n\n')

    flog.write('final summary: \n')
    flog.write(f'Data Info: mean {data_mean} std {data_std}\n')
    flog.write(f'Estimated(Final) Drift {generator.trainable_variables[0].numpy()[0]} Diffusion {generator.trainable_variables[1].numpy()[0]}\n')
    flog.write(f'Estimated(BestVal) Drift {best_val_drift} Diffusion {best_val_diffusion}\n\n')
    flog.close()





def get_args():
    parser = argparse.ArgumentParser()
    # Data processing
    parser.add_argument('--batch_size', nargs = '?', default = 128,  type = int, help =  'Batch size.')
    parser.add_argument('--val_included', dest = 'val_included', action = 'store_true', help = 'whether to include validation set')
    parser.add_argument('--val_not_included', dest = 'val_included', action = 'store_false', help = 'whether to include validation set')
    parser.set_defaults(val_included=True)
    # Whether to use synthetic data
    parser.add_argument('--synthetic', dest = 'synthetic', action = 'store_true', help = 'whether to use the synthetic dataset')
    parser.add_argument('--real_data', dest = 'synthetic', action = 'store_false', help = 'whether to use the synthetic dataset')
    parser.set_defaults(synthetic=True)
    parser.add_argument('--exp_no', nargs = '?', type = int, help = 'index of the rerun no')

    # Architectural hyperparameters. These are quite small for illustrative purposes.
    parser.add_argument('--init_noise_size', nargs = '?', default = 1, type = int, help = 'How many noise dimensions to sample at the start of the SDE.')
    parser.add_argument('--noise_size', nargs = '?', default = 1, type = int, help = 'How many dimensions the Brownian motion has.')
    parser.add_argument('--hidden_size', nargs = '?', default = 1, type = int, help='How big the hidden size of the generator SDE and the discriminator CDE are.')
    parser.add_argument('--mlp_size', nargs = '?', default = 1, type = int, help = 'How big the layers in the various MLPs are.')
    parser.add_argument('--num_layers', nargs = '?', default = 1, type = int, help = 'How many hidden layers to have in the various MLPs.')

    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    parser.add_argument('--generator_lr', nargs = '?', default = 1e-1, type = float, help ='Learning rate often needs careful tuning to the problem.')
    parser.add_argument('--discriminator_lr', nargs = '?', default = 1e-1, type = float, help = 'Learning rate often needs careful tuning to the problem.')
    parser.add_argument('--epochs', nargs = '?', default = 400,  type = int, help = 'How many steps to train both generator and discriminator for.')
    parser.add_argument('--weight_decay', nargs = '?', default = 0.0, type = float, help = 'Weight decay.')
    
    # Evaluate and save
    parser.add_argument('--save_path', nargs = '?', default = 'logs', type = str, help = 'log path')
    parser.add_argument('--model_path', nargs = '?', default = 'models', type = str, help = 'model path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
    
