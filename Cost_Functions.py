import numpy as np

# Class that contains the cost functions    
class CrossEntropyCost(object):
    # Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        #Calculates the cost of the output layer.
        return np.sum(np.nan_to_num(-y*np.log(x)-(1-y)*np.log(1-x)))
class QadraticCost(object):
    # Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        # Calculates the cost of the output layer.
        return 0.5*np.linalg.norm(x-y)**2