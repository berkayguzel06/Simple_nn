import numpy as np

class Loss:
    def remember_trainable_layers(self, trainable_layers):
            self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        losses = self.forward(output, y)
        data_loss = np.mean(losses)
        self.accumulated_sum += np.sum(losses)
        self.accumulated_count += len(losses)
        return data_loss
    
    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Categorical_Cross_Entropy(Loss):
    def forward(self, prediction, target):
        samples = len(prediction)
        y_clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        if len(target.shape) == 1:
            correctConfidences = y_clipped[range(samples), target]
        elif len(target.shape) == 2:
            correctConfidences = np.sum(y_clipped * target, axis=1)
        log = -np.log(correctConfidences)
        return log
    
    def backward(self, dvalues, target):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(target.shape) == 1:
            target = np.eye(labels)[target]
        self.dinputs = -target / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Binary_Cross_Entropy(Loss):
    def forward(self, predictions, target):
        y_pred_clip = np.clip(predictions, 1e-7, 1 - 1e-7)
        sample_loss = -(target * np.log(y_pred_clip) + (1 - target) * np.log(1 - y_pred_clip))
        sample_loss = np.mean(sample_loss, axis=-1)
        return sample_loss

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clip_dvalue = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(target / clip_dvalue - (1 - target) / (1 - clip_dvalue)) / outputs
        self.dinputs = self.dinputs / samples


class Mean_Square_Error(Loss):
    def forward(self, predictions, target):
        sample = np.mean((target - predictions) ** 2, axis=-1)
        return sample

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (target - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Mean_Absolute_Error(Loss):
    def forward(self, predictions, target):
        sample = np.mean(np.abs(target - predictions), axis=-1)
        return sample

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(target - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Log_Loss(Loss):
    def loss():
        pass

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

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        acc = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return acc
    
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    def addPrecision(self, y, reinit=False):
        if self.precision == None or reinit:
            self.precision = np.std(y) / 250
    def compare(self, predictions, y):
        return np.absolute(predictions-y) < self.precision
    
class Accuracy_Categorical(Accuracy):
    def addPrecision(self, y):
        pass
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    
class SGD():
    def __init__(self, lr=1., decay=0., momentum=0.):
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
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
        self.iterations += 1

class AdaGrad():
    def __init__(self, lr=1., decay=0., momentum=0., epsilon=1e-7):
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2  

        layer.weights += -self.currentlr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentlr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class RMSprop():
    def __init__(self, lr=0.001, decay=0., momentum=0., epsilon=1e-7, cmdr=0.9):
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.cmdr = cmdr

    def pre_update_params(self):
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.cmdr * layer.weight_cache + (1 - self.cmdr) * layer.dweights**2
        layer.bias_cache = self.cmdr * layer.bias_cache + (1 - self.cmdr) * layer.dbiases**2

        layer.weights += -self.currentlr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentlr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Adam():
    def __init__(self, lr=0.001, decay=0., momentum=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.currentlr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.currentlr = self.lr * (1. / (1.+ self.iterations * self.decay))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        # parameter update + normalization
        layer.weights += -self.currentlr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.currentlr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1