import numpy as np
import Cost_Functions
import Activation_Functions

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
    
    def fit(self, input, test=None, learning_rate=0.01, batch_size=1, epochs=10):
        # Forward pass through each layer
        for epoch in range(epochs):
            for batch_start in range(0, len(input), batch_size):
                batch_input = input[batch_start:batch_start + batch_size]
                for i in layers:
                    nabla_b = [np.zeros(b.shape) for b in i.biases]
                    nabla_w = [np.zeros(w.shape) for w in i.weights]
                    print(i.weights, i.biases)
                    i.forward(input,test)    
                    for x,y in batch_input:
                        weight_grad, bias_grad = i.backprop(x, y)
                        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, weight_grad)]
                        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, bias_grad)]
                    i.weights = [w-(learning_rate/len(batch_input))*nw
                        for w, nw in zip(i.weights, nabla_w)]
                    i.biases = [b-(learning_rate/len(batch_input))*nb
                       for b, nb in zip(i.biases, nabla_b)]
                    print(i.weights, i.biases)
                    input = i.output

class Layer():
    def __init__(self, inputs, neurons, cost='CrossEntropyCost', activation='sigmoid') -> None:
        # Initialize a layer with specified inputs, neurons, cost, and activation functions
        self.cost = helper_functions.get(cost)
        self.activation = helper_functions.get(activation)
        self.inputs = inputs
        
        # Initialize biases and weights with random values
        #self.biases = 0.01 * np.random.randn(1, neurons)
        self.biases = np.zeros((1, neurons))
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        
        # Add the current layer to the global list of layers
        layers.append(self)
    def backprop(self, x, y=None):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation.calculate(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost.cost_derivative(activations[-1], y) * \
                self.activation.calculate_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.inputs):
            z = zs[-l]
            sp = self.activation.calculate_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)
    def forward(self, input, y=None):
        # Perform the forward pass through the current layer
        self.output = self.activation.calculate(np.dot(input, self.weights) + self.biases)
        self.cost_value = self.cost.cost(self.output, y)