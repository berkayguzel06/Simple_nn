import numpy as np

class SGD():
    def __init__(self, lr=1., decay=0., momentum=0.):
        # Initialize SGD optimizer with learning rate, decay, and momentum
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        # Adjust learning rate based on decay, if specified
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        # Update weights and biases using SGD with momentum
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_update = self.momentum * layer.weight_momentums - self.currentlr * layer.dweights
            layer.weight_momentums = weight_update

            bias_update = self.momentum * layer.bias_momentums - self.currentlr * layer.dbiases
            layer.bias_momentums = bias_update
        else:
            weight_update = -self.currentlr * layer.dweights
            bias_update = -self.currentlr * layer.dbiases

        layer.weights += weight_update
        layer.biases += bias_update

    def post_update_params(self):
        # Increment iteration count
        self.iterations += 1

class AdaGrad():
    def __init__(self, lr=1., decay=0., momentum=0., epsilon=1e-7):
        # Initialize AdaGrad optimizer with learning rate, decay, momentum, and epsilon
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon

    def pre_update_params(self):
        # Adjust learning rate based on decay, if specified
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        # Update weights and biases using AdaGrad
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2  

        layer.weights += -self.currentlr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentlr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        # Increment iteration count
        self.iterations += 1

class RMSprop():
    # Difference between AdaGrad and RMSprop is, RMSprop provides lr decreases more smoothly.
    # cmdr = Cache memory decay rate
    def __init__(self, lr=0.001, decay=0., momentum=0., epsilon=1e-7, cmdr=0.9):
        # Initialize RMSprop optimizer with learning rate, decay, momentum, epsilon, and cache memory decay rate
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.cmdr = cmdr

    def pre_update_params(self):
        # Adjust learning rate based on decay, if specified
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        # Update weights and biases using RMSprop
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.cmdr * layer.weight_cache + (1 - self.cmdr) * layer.dweights**2
        layer.bias_cache = self.cmdr * layer.bias_cache + (1 - self.cmdr) * layer.dbiases**2

        layer.weights += -self.currentlr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentlr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        # Increment iteration count
        self.iterations += 1

class Adam():
    # cmdr = Cache memory decay rate
    def __init__(self, lr=0.001, decay=0., momentum=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        # Initialize Adam optimizer with learning rate, decay, momentum, epsilon, and beta parameters
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        # Adjust learning rate based on decay, if specified
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        # Update weights and biases using Adam optimizer
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square-rooted cache
        layer.weights += -self.currentlr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.currentlr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
