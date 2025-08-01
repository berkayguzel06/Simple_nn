# Simple Neural Network (Simplenn)

[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

![output](https://github.com/berkayguzel06/Simple_nn/assets/98205992/7ccf776f-dcf1-4720-9c15-7b14fafda42e)

A simple neural network implementation using Python.

## Overview

This repository contains a straightforward neural network (NN) implemented in Python. The neural network is designed for simplicity and serves for understanding the basics of neural networks, including layers, activations, loss functions, and optimizers.

## Features

- Customizable neural network architecture
- Various activation functions and loss functions
- Different optimizers (e.g., SGD, Adam, AdaGrad)
- Easy-to-use interface for training, evaluating, and predicting

## Prerequisites

- Python 3.x

## Installation

1. Clone the repository:

   ```bash
   https://github.com/berkayguzel06/Simple_nn.git
   cd simplenn

## Usage
1. Import the Model class and other necessary components from the simplenn package:
   ```python
   from simplenn.model import Model
   from simplenn.dense import Layer_Dense
   from simplenn.engine import ReLU, Softmax, Accuracy_Categorical, Categorical_Cross_Entropy, Adam
   ```
2. Create an instance of the Model class and define your neural network architecture:

   ```python
   model = Model()
   model.add(Layer_Dense(input_size, 16))
   model.add(ReLU())
   model.add(Layer_Dense(16, 3))
   model.add(Softmax())
   ```

3. Set the loss function, optimizer, and accuracy metric:
   ```python
   model.set(
      loss=Categorical_Cross_Entropy(),
      optimizer=Adam(lr=0.005, decay=1e-7),
      accuracy=Accuracy_Categorical()
   )
   ```
4. Finalize the model and train it on your data:
   ```python
   model.finalize()
   model.train(data, target, epochs=1000, print_every=100, validation=(validation_data, validation_target))
   ```
## Examples
Check the nnDemo.py script for an example of using the simplenn library on classification problems.
