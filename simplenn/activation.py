import numpy as np

class ReLU():
    def forward(self, inputs, training):
        # Save the input values for later use in the backward pass
        self.inputs = inputs
        # Apply ReLU activation element-wise
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Copy the gradient of the loss with respect to the output
        self.dinputs = dvalues.copy()
        # Set gradients to zero where the corresponding input was less than or equal to zero
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Softmax():
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs, training):
        # Exponentiate each input value and normalize to get probabilities
        self.inputs = inputs
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        # Save the calculated probabilities for later use in the backward pass
        self.output = probabilities

    def backward(self, dvalues):
        # Initialize an array to store the gradients with respect to inputs
        self.dinputs = np.empty_like(dvalues)
        
        # Loop over each example's output and gradient
        for idx, (single_out, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_out = single_out.reshape(-1, 1)
            
            # Calculate the Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_out) - np.dot(single_out, single_out.T)
            
            # Calculate the gradient with respect to inputs using the chain rule
            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvalues)

class Sigmoid():
    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Linear():
    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
