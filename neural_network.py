import numpy as np
import Cost_Functions
import Activation_Functions
from time import sleep
import nnfs

nnfs.init()

layers = []
# Define helper functions for cost and activation functions
helper_functions = {'CrossEntropyCost': Cost_Functions.CrossEntropyCost,
                    'QuadraticCost': Cost_Functions.QuadraticCost,
                    'MeanSquaredErrorCost': Cost_Functions.MeanSquaredErrorCost,
                    'sigmoid': Activation_Functions.SigmoidFunction,
                    'step': Activation_Functions.StepFunction,
                    'relu': Activation_Functions.ReluFunction,
                    'softmax': Activation_Functions.SoftmaxFunction}

class Network():
    def __init__(self) -> None:
        pass
    
    def fit(self, input, y, learning_rate=0.05, batch_size=1, epochs=10):
        first_input = input
        # Forward pass through each layer
        for epoch in range(epochs):
            for l in layers:
                l.forward(input,y)
                input = l.output
            input = first_input
            pred = np.argmax(layers[-1].output, axis=1)
            accuracy = np.mean(pred==y)
            loss = layers[-1].loss
            print('Iteration:',epoch,'loss:',loss,'acc:',accuracy)
            # Backpropagation - Compute gradients and update weights    
            delta = layers[-1].cost.delta(layers[-1].output,y) * layers[-1].activation.gradient(layers[-1].output)
            for i in range(len(layers)-1, 0, -1):
                delta = (layers[i].output-y)*2
                grad_z = layers[i].activation.gradient(layers[i].z)
                prev_out = layers[i-1].output
                gradient_weights = prev_out*grad_z*delta
                gradient_biases = np.sum(delta, axis=0, keepdims=True)
                layers[i].weights -= learning_rate * gradient_weights
                layers[i].biases -= learning_rate * gradient_biases

            

class Layer():
    def __init__(self, inputs, neurons, cost='CrossEntropyCost', activation='sigmoid') -> None:
        # Initialize a layer with specified inputs, neurons, cost, and activation functions
        self.cost = helper_functions.get(cost)
        self.activation = helper_functions.get(activation)
        self.neurons = neurons
        # Initialize biases and weights with random values
        #self.biases = 0.01 * np.random.randn(1, neurons)
        self.biases = np.zeros((1, neurons))
        self.weights = np.random.randn(inputs, neurons)
        # Add the current layer to the global list of layers
        layers.append(self)
    
    def forward(self, input, y=None):
        # Perform the forward pass through the current layer
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.activation.calculate(self.z)
        self.loss = self.cost.cost(self.output, y)