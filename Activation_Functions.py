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
    def sigmoid_prime(z):
        # Derivative of the sigmoid function.
        return SigmoidFunction.calculate(z)*(1-SigmoidFunction.calculate(z))

class ReluFunction(object):
    # Return the relu function.
    def calculate(x):
        return max(0,x)
    
class SoftmaxFunction(object):
    # Return the softmax function.
    def calculate(x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)