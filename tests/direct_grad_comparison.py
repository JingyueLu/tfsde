import torch
import tensorflow as tf
from sdegan.sde_gan import Generator, Discriminator
import os


'''
    Loading torch model into tensorflow and test the whether 
    the computed grad is the same as that of the torch model
'''


def tf_torch_compare(path = 'tests/torch_model_info/', batch_size =2, tol=1e-4):

    # NOTE: for this work, we have to remove all the random factor
    #       by seting the same fixed values for bm and init_noise
    
    print("Warning: This test should only be conducted when all random factors are
          removed.") 
    print("In this case, both models should  use the same fixed values for bm and init_noise.")

    # initilize the model
    model_info = torch.load(path+'model_info.pth')
    
    data_size = model_info['data_size']; initial_noise_size = model_info['initial_noise_size']
    noise_size = model_info['noise_size']; hidden_size = model_info['hidden_size']
    mlp_size = model_info['mlp_size']; num_layers = model_info['num_layers']

    ts = model_info['ts']
    ts = tf.constant(ts)
    
    generator = Generator(data_size, initial_noise_size, noise_size, 
                          hidden_size, mlp_size, num_layers)

    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers)
    
    sample_init = generator([ts, batch_size])
    score_init = discriminator(sample_init, generated = False)

    # FORWARD CHECK

    # we first load weights of the torch model
    ## Generator
    torch_generator_weights = torch.load(path + 'generator.pth')
    load_weight(torch_generator_weights, generator)
    ## Discriminator
    torch_discriminator_weights = torch.load(path + 'discriminator.pth')
    load_weight(torch_discriminator_weights, discriminator)


    # Assert tf models outputs are consistent with torch models
    ## Generator
    torch_output = torch.load(path + 'output.pth')
    torch_generated_samples = torch_output['generated_samples'].cpu().numpy()
    tf_generated_samples = generator([ts, 2]).numpy()

    assert abs(tf_generated_samples-torch_generated_samples).mean() < tol, 'Generator Forward check fails'
    
    ## Discriminator
    batch_data = tf.constant(torch.load(path+'batch_data.pth').cpu().numpy())
    tf_generated_score = discriminator(tf_generated_samples, generated=False).numpy()
    tf_real_score = discriminator(batch_data, generated=False).numpy()
    
    torch_generated_score = torch_output['generated_score'].cpu().numpy()
    torch_real_score = torch_output['real_score'].cpu().numpy()
    assert abs(tf_generated_score-torch_generated_score) < tol, 'Discriminator Forward check fails'
    assert abs(tf_real_score-torch_real_score) < tol, 'Discriminator Forward check fails'


    # BACKWARD CHECK
    with tf.GradientTape() as tape:

        generated_samples = generator([ts, batch_size])
        generated_score = discriminator(generated_samples, generated=True)
        real_score = discriminator(batch_data, generated=False)
        loss = generated_score - real_score
       
    tf_g_gradients, tf_d_gradients = tape.gradient(loss, [generator.trainable_variables, discriminator.trainable_variables])

    torch_g_gradients = torch.load(path+'generator_grad.pth')
    torch_d_gradients = torch.load(path+'discriminator_grad.pth')

    ## Generator
    for i in range(len(tf_g_gradients)):
        layer_g_gradient = tf_g_gradients[i]
        if len(layer_g_gradient.shape)==2:
            layer_g_gradient = tf.transpose(layer_g_gradient)
        assert abs(layer_g_gradient.numpy()-torch_g_gradients[i]).mean() < tol, 'Generator Backward check fails'

    ## Discriminator
    for i in range(len(tf_d_gradients)):
        layer_d_gradient = tf_d_gradients[i]
        if len(layer_d_gradient.shape)==2:
            layer_d_gradient = tf.transpose(layer_d_gradient)
        assert abs(layer_d_gradient.numpy()-torch_d_gradients[i]).mean() < tol, 'Discriminator Backward check fails'

    print('Tensorflow-implemented Generator and Discriminator passed both Forward and Backward tests')
    



def load_weight(weights_dict, model):
    '''
    load pytorch weights to tf models
    '''
    torch_values = list(weights_dict.values())
    idx = 0
    for layer in model.layers:
        torch_weights = []
        for layer_weight in layer.get_weights():
            if len(layer_weight.shape) == 2:
                torch_layer_weight = torch_values[idx].T.cpu().numpy()
                assert torch_layer_weight.shape == layer_weight.shape
                torch_weights.append(torch_layer_weight)

            elif len(layer_weight.shape) ==1:
                torch_layer_weight = torch_values[idx].cpu().numpy()
                assert torch_layer_weight.shape == layer_weight.shape
                torch_weights.append(torch_layer_weight)

            else:
                raise ValueError('Layer type not supported')
            idx += 1

        layer.set_weights(torch_weights)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_torch_compare()
