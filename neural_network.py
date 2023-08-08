import numpy as np
import Cost_Functions
import Activation_Functions

np.random.seed(0)

layers = []

# Define helper functions for cost and activation functions
helper_functions = {'CrossEntropyCost': Cost_Functions.CrossEntropyCost,
                    'QuadraticCost': Cost_Functions.QuadraticCost,
                    'SigmoidFunction': Activation_Functions.SigmoidFunction,
                    'StepFunction': Activation_Functions.StepFunction,
                    'ReluFunction': Activation_Functions.ReluFunction,
                    'SoftmaxFunction': Activation_Functions.SoftmaxFunction}

class Network():
    def __init__(self) -> None:
        pass
    
    def fit(self, input):
        # Forward pass through each layer
        for i in layers:
            input = i.forward(input)
            print(input)

class Layer():
    def __init__(self, inputs, neurons, cost='CrossEntropyCost', activation='SigmoidFunction') -> None:
        # Initialize a layer with specified inputs, neurons, cost, and activation functions
        self.cost = helper_functions.get(cost)
        self.activation = helper_functions.get(activation)
        self.inputs = inputs
        
        # Initialize biases and weights with random values
        self.biases = 0.1 * np.random.randn(1, neurons)
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        
        # Add the current layer to the global list of layers
        layers.append(self)
    
    def forward(self, input):
        # Perform the forward pass through the current layer
        self.output = self.activation.calculate(np.dot(input, self.weights) + self.biases)
        return self.output