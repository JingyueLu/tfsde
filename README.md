# tfsde
This repository provides tensorflow implementation of SDE-GAN. SDE-GAN is first proposed in [Neural SDEs as Inifinite-Dimensional GANs](https://arxiv.org/abs/2102.03657) and further developed in [Efficient and Accurate Gradients for Neural SDEs](https://arxiv.org/abs/2105.13493). The original library is in Pytorch.

## Directories

* **tfsde** corresponds to the [**torchsde**](https://github.com/google-research/torchsde) directory in the original library. The original library supports much wider applications with various SDE solvers. Since our focus here is Reversible Heun method, we have removed all other SDE solvers and their related supporting funcitonality.  
* **tfcde** is adopted from the [**torchcde**](https://github.com/patrick-kidger/NeuralCDE). It is used in the discriminator for constructing neural controlled differential equations. Similarly, for the purpose of this project, we have removed all other interpolation methods and kept the linear interpolation one only.
* **sdegan** is the main directory for SDE-GAN. It contains the script describing the model architecture of SDE-GAN and the training script. We have also stored the HSI data in the directory.
* **scripts** contains running bash scripts. User can simply modify the model parameters in sdegan.sh and run the experiement as
```bash
./scripts/sdegan.sh
```
* **tests** collects all tests we have done to ensure our re-implementation is correct. More details below.
* **model** stores all trained models
* **logs** records all necessary training information and related plots can be visulised in tensorboard by running
```bash
tensorboard ---logdir logs
```

## Tests
We give more details of the tests we have done to ensure our re-implementation is correct. The model consists of a random part and a deterministic part. The random part requires sampling Browninian motion. To ensure our re-implementation satisfies all required sampling properties, we also re-implemented the original browninan_interval test file in tensorflow. We run tests for brownian motion implementation as follows:
```bash
cd tests
# using pytest package to run all tests
pytest
cd ..
```
For the deterministic part, we need to check model implementation (Forward pass) and gradients computation (Backward propagation). To do so, we take a direct approach. We ran the origional pytorch library and, at a random point, record all model weights, initial random noise, brownian motion object, and all computed gradients. We then load the weights into the tensorflow model and set random components(initial random noise and brownian motion object) to be the same. We compare the forward computation results and backward gradients results with the recorded results of the pytorch model. The results should be the same.
```bash
vim tfsde/_core/methods/reversible_heun.py
# In the reversible_heun.py file, uncomment all parts headed with Testing

python tests/direct_grad_comparison.py
```




