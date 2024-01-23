import numpy as np

class ReLU():
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Softmax():
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs, training):
        self.inputs = inputs
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for idx, (single_out, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_out = single_out.reshape(-1, 1)      
            jacobian_matrix = np.diagflat(single_out) - np.dot(single_out, single_out.T)      
            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvalues)

class Sigmoid():
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Linear():
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs
