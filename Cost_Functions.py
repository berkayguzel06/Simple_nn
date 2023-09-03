import numpy as np

# Class that contains the cost functions    
class CrossEntropyCost(object):
    # Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        #Calculates the cost of the output layer.
        size = len(x)
        epsilon = 1e-7
        x = np.clip(x, epsilon, 1 - epsilon)
        z = np.log(x)
        return -np.sum(x*np.log(x))/size
    def delta(x,y):
        # Derivative of the cross-entropy cost function.
        return (x-y)
    
class QuadraticCost(object):
    # Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        # Calculates the cost of the output layer.
        return 0.5*np.linalg.norm(x-y)**2
    
class MeanSquaredErrorCost(object):
    # Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        # Calculates the cost of the output layer.
        return np.mean((np.array(x)-np.array(y))**2)
    def delta(x,y):
        # Derivative of the mean squared error cost function.
        return 2*(x-y)/len(y)
    
class MeanAbsoluteError(object):
    def cost(x,y):
        return np.mean(np.abs(np.array(x)-np.array(y)))