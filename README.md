# dae
##Deep Autoencoder module for python

## Deep Autoencoder (DAE) Module
This is a Python module that provides functionality for creating and training deep autoencoders. Autoencoders are a type of neural network that can be used for unsupervised learning and dimensionality reduction. They are commonly used for feature extraction, anomaly detection, and data compression tasks.

## What is an Autoencoder?
An autoencoder is a neural network architecture that consists of two main parts: an encoder and a decoder. The encoder takes an input and maps it to a lower-dimensional representation called the latent space. The decoder then takes this lower-dimensional representation and reconstructs the original input as closely as possible. The encoder and decoder are trained together in an end-to-end manner to minimize the reconstruction error.

Autoencoders are useful for various applications, such as image and text data. They can learn meaningful representations by capturing the most important features of the input data in the latent space. These learned representations can then be used for tasks such as data compression, denoising, and dimensionality reduction.

## About this Autoencoder Module
The dae.py module provided here is specifically designed for creating and training deep autoencoders for airfoil shape optimization. However, it can be easily adapted and used for other applications as well. It leverages the Keras library to define and train the autoencoder model.

## Installation
To use this module, you need to have the following dependencies installed:

Keras (version 2.0 or higher)
NumPy
Matplotlib
scikit-learn
You can install these dependencies using pip:

`pip install keras numpy matplotlib scikit-learn`


Once you have installed the dependencies, you can download the dae.py file from this repository and include it in your project.

## Usage
Here's an example of how to use the dae.py module to train a deep autoencoder:

```ruby
import numpy as np
from autoencoder import dae

# Set hyperparameters
n_epochs = 100
batch_size = 32
activations = 'relu'
learning_rate = 0.001
nodes = [128, 64, 32]
patience = 5

# Path to the shapes file
shapes_path = 'shapes.csv'

# Number of nodes in the central layer
nodes_central = 16

# Build and train the autoencoder
track, dataset, X_train, X_val, X_test, X_train_unscaled, X_val_unscaled, predicted_val, predicted_val_unscaled, mse_val, nmse_val, var_orig, mean_shape = dae.dae_build(shapes_path, [n_epochs, batch_size, activations, learning_rate, nodes, patience], nodes_central)

# Access the results
print("MSE:", mse_val)
print("NMSE:", nmse_val)
```

Make sure to replace 'shapes.csv' with the actual path to your shapes file.

Contributing
If you'd like to contribute to this module, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your feature or make changes to fix the bug.
4. Write tests to cover your changes (if applicable).
5. Commit your changes and push them to your fork.
6. Submit a pull request explaining your changes.

License
This module is released under the MIT License. See LICENSE for more information.
