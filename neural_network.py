import numpy as np
import random as rnd

class CrossEntropyCost(object):
    #Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        #Calculates the cost of the output layer.
        return np.sum(np.nan_to_num(-y*np.log(x)-(1-y)*np.log(1-x)))

class QadraticCost(object):
    #Return the cost associated with an output ``x`` and desired output ``y``.
    def cost(x,y):
        #Calculates the cost of the output layer.
        return 0.5*np.linalg.norm(x-y)**2


class Network():
    def __init__(self,size,cost=CrossEntropyCost) -> None:
        """
        The list ``sizes`` contains the number of neurons in the respective layers of the network.
        For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons,
        the second layer 3 neurons, and the third layer 1 neuron. The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1. Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers.
        """
        self.layers = len(size)
        self.default_weight_initializer()
        self.size = size
        self.cost = cost

    def feedforward(self,input):
        for w,b in zip(self.biase, self.weights):
            input = sigmoid(np.dot(w,input)+b)
        return input
    
    def SDG(self,train,epochs,batch_size,eta,test=None):
        #Train the neural network using mini-batch stochastic gradient descent.
        if test: test_size = len(test)
        n = len(train)
        for i in range(epochs):
            rnd.shuffle(train)
            batches = [train[k:k in batch_size] for k in range(0,n,batch_size)]
            for batch in batches:
                self.update_batch(batch,eta)
            if test:
                print('Epoch {0}: {1} {2}'.format(i,self.evaluate(test),test_size))
            else:
                print('Epoch {0} complete'.format(i))
    
    def update_batch(self, batch, eta):
        #Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        nabla_b = [np.zeros(i) for i in self.biase]
        nabla_w = [np.zeros(i) for i in self.weights]

        for x,y in batch:
            delta_b, delta_w = self.backprop(x,y)

            nabla_b = [nb + db for nb,db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw,dw in zip(nabla_w, delta_w)]

        self.weight = [w - (eta-len(batch))*nw for w,nw in zip(self.weight, nabla_w)]
        self.biase = [b - (eta-len(batch))*nb for b,nb in zip(self.biase, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biase`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biase]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biase, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sp
            nabla_b[-l] = delta
            
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def default_weight_initializer(self):
        #Initialize each weight using a Gaussian distribution with mean 0, and variance 1 
        #over the square root of the number of weights connecting to the same neuron.
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.size[:-1], self.size[1:])]

    def large_weight_initializer(self):
        #Initialize the weights using a Gaussian distribution with mean 0, and variance 1.
        self.biase = [np.random.randn(y,1) for y in self.size[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.size[:-1],self.size[1:])]


def sigmoid(z):
    #The sigmoid function.
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    #Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z))