# Adversarial Autoencoder

This repository contains a PyTorch implementation of an adversarial autoencoder (AAE). An AAE is a type of generative model that can be used for unsupervised learning. It is trained to encode input data as a low-dimensional latent code, and then decode the latent code to recreate the original input data. The AAE also includes an adversarial component, which involves training a discriminator to differentiate between the encoded latent codes and a prior distribution (e.g. a Gaussian distribution).

## Requirements
To use the code in this repository, you will need to have PyTorch installed. The code is written in Python 3 and has been tested with PyTorch 1.7.0.

## Usage
To train an AAE, you can use the train.py script. This script takes a number of arguments, including the path to the training data, the dimensions of the latent code, and the number of epochs to train for. For example, to train an AAE with a 2-dimensional latent code on the MNIST dataset for 100 epochs, you could run the following command:

```python
python train.py --data-path=./data/mnist.pkl --z_dim=2 --epochs=100 --lambda=0.99
```

where lambda is the weight of the reconstruction loss. The default value is 0.99.

This will train the model and generate samples from the trained AAE and display them as images. The images and logs will be saved in the ./AAE_logs directory by default (you can change this by passing the --log_dir argument to the script). 

The logs can be visualized using TensorBoard. To do this, run the following command:

```python
tensorboard --logdir=./AAE_logs
```

## Contributions
We welcome contributions to this repository. If you would like to contribute, please fork this repository and submit a pull request.

## References
Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015).