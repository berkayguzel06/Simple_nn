import numpy as np

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs, training):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    def get_parameters(self):
        return self.weights, self.biases
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout:
    def __init__(self, rate) -> None:
        self.rate = 1 - rate
    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.bin_mask = np.random.binomial(1, self.rate, self.inputs.shape) / self.rate
        self.output = inputs * self.bin_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.bin_mask