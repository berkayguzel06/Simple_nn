import numpy as np
# Class that contains the activation functions
class StepFunction(object):
    # Return the step function.
    def calculate(x):
        return 1 if x>=0 else 0

class SigmoidFunction(object):
    # Return the sigmoid function. 
    def calculate(x):
        return 1/(1+np.exp(-x))
    def gradient(z):
        # Derivative of the sigmoid function.
        return SigmoidFunction.calculate(z)*(1-SigmoidFunction.calculate(z))

class ReluFunction(object):
    # Return the relu function.
    def calculate(x):
        return np.maximum(0,x)
    def gradient(z):
        # Calculate the derivative of the ReLU function
        return np.where(z > 0, 1, 0)
    
class SoftmaxFunction(object):
    # Return the softmax function.
    def calculate(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    def gradient(z):
        # Calculate the derivative of the softmax function
        softmax = SoftmaxFunction.calculate(z)
        return softmax * (1 - softmax)